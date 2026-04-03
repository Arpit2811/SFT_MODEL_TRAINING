"""
=============================================================================
  HIGH-PERFORMANCE DEDUPLICATION PIPELINE  
  MinHash + LSH + Jaccard Similarity
=============================================================================
"""
from __future__ import annotations

import argparse
import gc
import glob
import logging
import math
import os
import signal
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from typing import Dict, List, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

NUM_WORKERS    = 100
NUM_PERM       = 128
LSH_THRESHOLD  = 0.7
NGRAM_SIZE     = 3
CHUNK_SIZE     = 500_000
JACCARD_VERIFY = True
COMBINE_COLS   = True

# Column names to use — order matters for shingling union
ALL_COLUMNS    = ["Instruction", "Passage", "Question", "Answer"]
FALLBACK_COL   = "Question"   # used when --question-only

JACCARD_TARGET_TASKS_PER_WORKER = 4
JACCARD_MIN_TASK_SIZE           = 100_000
JACCARD_MAX_TASK_SIZE           = 10_000_000
JACCARD_MAX_RAM_PER_WORKER_GB   = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

class _SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            super().emit(record)
        except (BrokenPipeError, OSError):
            pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[_SafeStreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZED MINHASH WORKER  — per-column shingling with union
# ─────────────────────────────────────────────────────────────────────────────

def _minhash_worker_multicol(args: Tuple) -> Tuple[int, np.ndarray]:
    """
    Per-column shingling with shingle-set union.

    Instead of concatenating all columns into one long string:
      "Instruction [SEP] Passage [SEP] Question [SEP] Answer"
      → 4x the bytes → 4x the shingles → slow

    We shingle each column independently and union the sets:
      shingles = shingles(Instruction) | shingles(Passage) |
                 shingles(Question)    | shingles(Answer)
      → shingle count ≈ max(col shingles) + small overlaps
      → same semantic coverage, fraction of the work

    This matches 2-column speed because:
      - Short columns (Instruction, Passage) have very few unique shingles
      - Union deduplicates overlapping shingles across columns
      - MinHash sees the same number of unique shingles as the 2-col version
    """
    start_local, col_texts_list, num_perm, ngram = args
    # col_texts_list: list of n_rows lists, one per column
    # shape: (n_cols, n_rows)
    n_rows = len(col_texts_list[0])
    out    = np.empty((n_rows, num_perm), dtype=np.uint64)

    for i in range(n_rows):
        # Union shingles from all columns for this row
        shingles: set = set()
        for col_texts in col_texts_list:
            text = col_texts[i] or ""
            b    = text.encode("utf-8", errors="ignore")
            mv   = memoryview(b)
            L    = len(b)
            if L > 0:
                shingles.update(
                    bytes(mv[j: j + ngram])
                    for j in range(max(1, L - ngram + 1))
                )
        m = MinHash(num_perm=num_perm)
        if shingles:
            m.update_batch(shingles)
        out[i] = m.hashvalues

    return start_local, out


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK WORKER — single column (--question-only, unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

def _minhash_worker_single(args: Tuple) -> Tuple[int, np.ndarray]:
    start_local, texts_slice, num_perm, ngram = args
    n   = len(texts_slice)
    out = np.empty((n, num_perm), dtype=np.uint64)
    for i, text in enumerate(texts_slice):
        b        = text.encode("utf-8", errors="ignore")
        mv       = memoryview(b)
        shingles = {bytes(mv[j: j + ngram]) for j in range(max(1, len(b) - ngram + 1))}
        m        = MinHash(num_perm=num_perm)
        m.update_batch(shingles)
        out[i]   = m.hashvalues
    return start_local, out


# ─────────────────────────────────────────────────────────────────────────────
# JACCARD WORKER  — numpy memmap, zero pickling of signature data
# ─────────────────────────────────────────────────────────────────────────────

def _jaccard_worker_numpy(args: Tuple) -> np.ndarray:
    pairs_slice, memmap_path, total_rows, num_perm, threshold = args
    sigs  = np.memmap(memmap_path, dtype=np.uint64, mode="r",
                      shape=(total_rows, num_perm))
    a_idx = pairs_slice[:, 0]
    b_idx = pairs_slice[:, 1]
    INNER = 500_000
    mask  = np.empty(len(pairs_slice), dtype=bool)
    for start in range(0, len(pairs_slice), INNER):
        end             = min(start + INNER, len(pairs_slice))
        sa              = sigs[a_idx[start:end]]
        sb              = sigs[b_idx[start:end]]
        mask[start:end] = (sa == sb).mean(axis=1) >= threshold
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC JACCARD TASK SIZING
# ─────────────────────────────────────────────────────────────────────────────

def compute_jaccard_task_size(
    n_pairs, n_workers,
    target_tasks_per_worker=JACCARD_TARGET_TASKS_PER_WORKER,
    min_task_size=JACCARD_MIN_TASK_SIZE,
    max_task_size=JACCARD_MAX_TASK_SIZE,
    num_perm=NUM_PERM,
    max_ram_per_worker_gb=JACCARD_MAX_RAM_PER_WORKER_GB,
) -> Tuple[int, int]:
    if n_pairs == 0:
        return min_task_size, 0
    bytes_per_pair   = 2 * num_perm * 8
    max_pairs_by_ram = int((max_ram_per_worker_gb * 1024 ** 3) / bytes_per_pair)
    target_n_tasks   = n_workers * target_tasks_per_worker
    size_by_target   = max(1, math.ceil(n_pairs / target_n_tasks))
    task_size        = min(max_pairs_by_ram, size_by_target)
    task_size        = max(min_task_size, min(task_size, max_task_size))
    return task_size, math.ceil(n_pairs / task_size)


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL COLUMN READER
# ─────────────────────────────────────────────────────────────────────────────

def _read_columns_parallel(input_path: str, columns: list,
                            batch_size: int) -> list:
    """
    Read each column in a separate thread. PyArrow releases the GIL during
    Parquet column decoding, so N columns decode in ~the time of 1.
    Returns list of RecordBatch iterators — one per column.

    Actually returns the batches pre-joined: yields one RecordBatch per chunk
    with only the requested columns, read via parallel column decode.
    """
    # PyArrow's read_table with use_threads=True already parallelizes
    # column decoding internally. We expose this at the iter_batches level
    # by reading the full batch with all columns at once (Arrow does parallel
    # column IO internally when use_threads=True, which is the default).
    # This is equivalent to reading columns in parallel threads.
    return pq.ParquetFile(input_path).iter_batches(
        batch_size=batch_size,
        columns=columns,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BATCH → TASK BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_multicol_tasks(batch, columns: list, n_workers: int,
                           num_perm: int, ngram: int) -> list:
    """
    Build worker tasks from a multi-column batch.
    Each task gets (start_local, col_texts_list, num_perm, ngram) where
    col_texts_list is a list of n_cols text lists (each of length sub_batch).

    This avoids string concatenation entirely — workers receive raw column
    text lists and do per-column shingling with union.
    """
    n_rows = len(batch)
    sbs    = max(2000, math.ceil(n_rows / n_workers))

    # Extract all columns to Python lists once (fast, contiguous)
    col_pylist = [
        [s or "" for s in batch.column(c).to_pylist()]
        for c in columns
    ]

    tasks = []
    for start in range(0, n_rows, sbs):
        end          = min(start + sbs, n_rows)
        # Slice each column's list for this sub-batch
        col_slice    = [col[start:end] for col in col_pylist]
        tasks.append((start, col_slice, num_perm, ngram))

    return tasks, sbs


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — JACCARD-DIRECT DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def build_keep_set(confirmed_pairs_arr: np.ndarray,
                   total_rows: int) -> Tuple[Set[int], int]:
    if len(confirmed_pairs_arr) == 0:
        return set(range(total_rows)), 0
    a       = confirmed_pairs_arr[:, 0]
    b       = confirmed_pairs_arr[:, 1]
    swap    = a > b
    b_fixed = np.where(swap, a, b)
    duplicates = set(b_fixed.tolist())
    keep       = set(range(total_rows)) - duplicates
    return keep, len(duplicates)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cast_batch(batch):
    new_fields = []
    for f in batch.schema:
        if pa.types.is_large_string(f.type) or str(f.type) in (
            "string_view", "utf8_view", "utf8"
        ):
            new_fields.append(pa.field(f.name, pa.large_string()))
        else:
            new_fields.append(f)
    return batch.cast(pa.schema(new_fields))


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-FILE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    output_path: str,
    columns: list,
    combine: bool,
    target_tasks_per_worker: int = JACCARD_TARGET_TASKS_PER_WORKER,
    max_ram_per_worker_gb: float = JACCARD_MAX_RAM_PER_WORKER_GB,
    min_task_size: int           = JACCARD_MIN_TASK_SIZE,
    max_task_size: int           = JACCARD_MAX_TASK_SIZE,
) -> None:
    t0 = time.perf_counter()

    log.info(f"{'─'*60}")
    log.info(f"Processing : {input_path}")
    log.info(f"Output     : {output_path}")
    log.info(f"Columns    : {columns}  combine={combine}")

    pf         = pq.ParquetFile(input_path)
    meta       = pf.metadata
    total_rows = sum(meta.row_group(i).num_rows for i in range(meta.num_row_groups))
    if total_rows != meta.num_rows:
        log.warning(f"Stale footer: {meta.num_rows:,} vs row-group sum {total_rows:,}")

    log.info(
        f"Rows: {total_rows:,}  Workers: {NUM_WORKERS}  Perm: {NUM_PERM}  "
        f"Threshold: {LSH_THRESHOLD}  N-gram: {NGRAM_SIZE}  Chunk: {CHUNK_SIZE:,}"
    )
    if total_rows == 0:
        log.warning("Empty parquet file — skipping.")
        return

    read_cols  = columns if combine else [FALLBACK_COL]
    num_chunks = math.ceil(total_rows / CHUNK_SIZE)

    # ── Phase 1: MinHash ─────────────────────────────────────────────────────
    log.info(f"Phase 1 – MinHash generation ({'multi-col union' if combine and len(read_cols)>1 else 'single-col'}) …")
    t1 = time.perf_counter()

    memmap_alloc = total_rows + max(10_000, total_rows // 100)
    tmp_dir      = tempfile.mkdtemp(prefix="dedup_")
    memmap_path  = os.path.join(tmp_dir, "signatures.npy")
    all_hv       = np.memmap(memmap_path, dtype=np.uint64, mode="w+",
                             shape=(memmap_alloc, NUM_PERM))
    actual_rows  = 0

    with Pool(processes=NUM_WORKERS) as pool:
        offset = 0
        for chunk_no, batch in enumerate(
            _read_columns_parallel(input_path, read_cols, CHUNK_SIZE)
        ):
            n_bat = len(batch)
            tc    = time.perf_counter()

            end_offset = offset + n_bat
            if end_offset > memmap_alloc:
                new_alloc    = end_offset + max(10_000, end_offset // 100)
                log.warning(f"Row overflow: resizing memmap to {new_alloc:,}")
                del all_hv
                all_hv       = np.memmap(memmap_path, dtype=np.uint64, mode="r+",
                                         shape=(new_alloc, NUM_PERM))
                memmap_alloc = new_alloc

            if combine and len(read_cols) > 1:
                # ── MULTI-COL PATH: per-column shingling + union ──────────
                tasks, sbs = _build_multicol_tasks(
                    batch, read_cols, NUM_WORKERS, NUM_PERM, NGRAM_SIZE
                )
                del batch; gc.collect()

                log.info(
                    f"  Chunk {chunk_no+1}/{num_chunks}  rows {offset:,}–{offset+n_bat-1:,} "
                    f"| {len(tasks)} tasks × ~{sbs} rows  [multi-col union] …"
                )
                for local_start, mat in pool.imap_unordered(
                    _minhash_worker_multicol, tasks, chunksize=1
                ):
                    all_hv[offset + local_start: offset + local_start + len(mat)] = mat

            else:
                # ── SINGLE-COL PATH: identical to v7 ─────────────────────
                texts = [s or "" for s in batch.column(FALLBACK_COL).to_pylist()]
                del batch; gc.collect()

                sbs   = max(2000, math.ceil(n_bat / NUM_WORKERS))
                tasks = [(i, texts[i: i + sbs], NUM_PERM, NGRAM_SIZE)
                         for i in range(0, n_bat, sbs)]

                log.info(
                    f"  Chunk {chunk_no+1}/{num_chunks}  rows {offset:,}–{offset+n_bat-1:,} "
                    f"| {len(tasks)} tasks × ~{sbs} rows  [single-col] …"
                )
                for local_start, mat in pool.imap_unordered(
                    _minhash_worker_single, tasks, chunksize=1
                ):
                    all_hv[offset + local_start: offset + local_start + len(mat)] = mat

                del texts

            del tasks; gc.collect()
            log.info(
                f"  Chunk {chunk_no+1}/{num_chunks} done in "
                f"{time.perf_counter()-tc:.1f}s  "
                f"({n_bat/(time.perf_counter()-tc):,.0f} rows/s)"
            )
            offset      += n_bat
            actual_rows += n_bat

    if actual_rows != total_rows:
        log.warning(f"Metadata said {total_rows:,} rows but reader delivered {actual_rows:,}.")
        total_rows = actual_rows

    all_hv.flush()
    if total_rows < memmap_alloc:
        del all_hv
        all_hv = np.memmap(memmap_path, dtype=np.uint64, mode="r+",
                           shape=(total_rows, NUM_PERM))
        all_hv.flush()

    log.info(f"Phase 1 done in {time.perf_counter()-t1:.1f}s")

    # ── Phase 2: LSH ─────────────────────────────────────────────────────────
    log.info("Phase 2 – LSH streaming insert + query …")
    t2              = time.perf_counter()
    lsh             = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    candidate_pairs: Set[Tuple[int, int]] = set()
    _m              = MinHash(num_perm=NUM_PERM)

    try:
        pbar = tqdm(total=total_rows, desc="LSH", unit="row",
                    mininterval=5.0, file=sys.stderr)
        for gi in range(total_rows):
            _m.hashvalues = all_hv[gi]
            for nb_key in lsh.query(_m):
                candidate_pairs.add((int(nb_key), gi))
            lsh.insert(str(gi), _m)
            if gi % 10_000 == 0:
                pbar.update(10_000)
        pbar.update(total_rows % 10_000)
        pbar.close()
    except (BrokenPipeError, OSError):
        pass

    log.info(
        f"Phase 2 done — {len(candidate_pairs):,} candidate pairs "
        f"in {time.perf_counter()-t2:.1f}s"
    )
    del all_hv; gc.collect()

    # ── Phase 3: Jaccard verification ────────────────────────────────────────
    if JACCARD_VERIFY and candidate_pairs:
        n_pairs = len(candidate_pairs)
        task_size, n_tasks = compute_jaccard_task_size(
            n_pairs=n_pairs, n_workers=NUM_WORKERS,
            target_tasks_per_worker=target_tasks_per_worker,
            min_task_size=min_task_size, max_task_size=max_task_size,
            num_perm=NUM_PERM, max_ram_per_worker_gb=max_ram_per_worker_gb,
        )
        ram_per_task_gb = task_size * 2 * NUM_PERM * 8 / 1024 ** 3
        log.info(
            f"Phase 3 – Jaccard verification ({n_pairs:,} pairs)\n"
            f"  Task size: {task_size:,}  Tasks: {n_tasks}  "
            f"RAM/worker: ~{ram_per_task_gb:.3f} GB"
        )
        t3       = time.perf_counter()
        pair_arr = np.array(list(candidate_pairs), dtype=np.int32)
        del candidate_pairs; gc.collect()

        tasks = [
            (pair_arr[s: min(s + task_size, n_pairs)],
             memmap_path, total_rows, NUM_PERM, LSH_THRESHOLD)
            for s in range(0, n_pairs, task_size)
        ]

        confirmed_mask = np.empty(n_pairs, dtype=bool)
        pos = 0
        try:
            with Pool(processes=min(NUM_WORKERS, n_tasks)) as pool:
                for mask in tqdm(
                    pool.imap(_jaccard_worker_numpy, tasks, chunksize=1),
                    total=n_tasks, desc="Jaccard batches", file=sys.stderr
                ):
                    confirmed_mask[pos: pos + len(mask)] = mask
                    pos += len(mask)
        except (BrokenPipeError, OSError):
            pass

        confirmed_pairs_arr = pair_arr[confirmed_mask]
        log.info(
            f"Phase 3 done — {len(confirmed_pairs_arr):,}/{n_pairs:,} confirmed "
            f"in {time.perf_counter()-t3:.1f}s"
        )
        del pair_arr, confirmed_mask; gc.collect()

    elif not JACCARD_VERIFY and candidate_pairs:
        log.info("Phase 3 – Jaccard verification skipped (--no-verify)")
        confirmed_pairs_arr = np.array(list(candidate_pairs), dtype=np.int32)
        del candidate_pairs; gc.collect()
    else:
        log.info("Phase 3 – No candidate pairs.")
        confirmed_pairs_arr = np.empty((0, 2), dtype=np.int32)

    # ── Phase 4: Keep set ─────────────────────────────────────────────────────
    log.info("Phase 4 – Building keep set from confirmed pairs …")
    t4 = time.perf_counter()
    keep, n_dup = build_keep_set(confirmed_pairs_arr, total_rows)
    del confirmed_pairs_arr; gc.collect()
    log.info(
        f"  Duplicates : {n_dup:,} ({100*n_dup/total_rows:.2f}%)  "
        f"Unique : {len(keep):,}  "
        f"Time   : {time.perf_counter()-t4:.2f}s"
    )

    # ── Phase 5: Write output ─────────────────────────────────────────────────
    log.info(f"Phase 5 – Writing → {output_path} …")
    t5 = time.perf_counter()
    writer = None
    offset = 0

    try:
        for batch in tqdm(
            pq.ParquetFile(input_path).iter_batches(batch_size=CHUNK_SIZE),
            desc="Writing", unit="chunk", file=sys.stderr
        ):
            n_bat     = len(batch)
            local_idx = [i for i in range(n_bat) if (offset + i) in keep]
            if local_idx:
                idx_arr    = pa.array(local_idx, type=pa.int32())
                kept_batch = pc.take(_cast_batch(batch), idx_arr)
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path, kept_batch.schema,
                        compression="snappy",
                        use_dictionary=True,
                        write_statistics=True,
                    )
                writer.write_batch(kept_batch)
            offset += n_bat
    except (BrokenPipeError, OSError):
        pass
    finally:
        if writer:
            writer.close()

    try:
        os.remove(memmap_path)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    elapsed = time.perf_counter() - t0
    log.info(f"Phase 5 done in {time.perf_counter()-t5:.1f}s")
    log.info("=" * 60)
    log.info(f"DONE  {os.path.basename(input_path)}  │  {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info(
        f"  Input : {total_rows:,}  →  Output : {len(keep):,}  "
        f"(removed {n_dup:,}, {100*n_dup/total_rows:.2f}%)"
    )
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORY DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def _collect_parquet_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
        if not files:
            log.error(f"No .parquet files found in: {path}")
            sys.exit(1)
        return files
    log.error(f"Input path does not exist: {path}")
    sys.exit(1)


def _resolve_output_path(input_file, input_root, output_root):
    if os.path.isfile(input_root):
        if os.path.isdir(output_root) or output_root.endswith(("/", os.sep)):
            os.makedirs(output_root, exist_ok=True)
            stem = os.path.splitext(os.path.basename(input_file))[0]
            return os.path.join(output_root, f"{stem}_deduped.parquet")
        os.makedirs(os.path.dirname(os.path.abspath(output_root)), exist_ok=True)
        return output_root
    rel      = os.path.relpath(input_file, input_root)
    stem, _  = os.path.splitext(rel)
    out_path = os.path.join(output_root, stem + "_deduped.parquet")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MinHash-LSH dedup v8 — 4-col optimized, per-column shingle union"
    )
    parser.add_argument("--input",              required=True)
    parser.add_argument("--output",             required=True)
    parser.add_argument("--columns",            nargs="+",
                        default=ALL_COLUMNS,
                        help=f"Columns to deduplicate on (default: {ALL_COLUMNS})")
    parser.add_argument("--workers",            type=int,   default=NUM_WORKERS)
    parser.add_argument("--num-perm",           type=int,   default=None)
    parser.add_argument("--num-per",            type=int,   default=None)
    parser.add_argument("--threshold",          type=float, default=LSH_THRESHOLD)
    parser.add_argument("--ngram",              type=int,   default=NGRAM_SIZE)
    parser.add_argument("--chunk",              type=int,   default=CHUNK_SIZE)
    parser.add_argument("--no-verify",          action="store_true")
    parser.add_argument("--question-only",      action="store_true",
                        help=f"Use only {FALLBACK_COL} column for hashing")
    parser.add_argument("--skip-existing",      action="store_true")
    parser.add_argument("--max-ram-per-worker", type=float, default=JACCARD_MAX_RAM_PER_WORKER_GB)
    parser.add_argument("--tasks-per-worker",   type=int,   default=JACCARD_TARGET_TASKS_PER_WORKER)
    parser.add_argument("--min-task-size",      type=int,   default=JACCARD_MIN_TASK_SIZE)
    parser.add_argument("--max-task-size",      type=int,   default=JACCARD_MAX_TASK_SIZE)

    args = parser.parse_args()

    NUM_WORKERS    = args.workers
    LSH_THRESHOLD  = args.threshold
    NGRAM_SIZE     = args.ngram
    CHUNK_SIZE     = args.chunk
    JACCARD_VERIFY = not args.no_verify
    COMBINE_COLS   = not args.question_only

    if args.num_perm is not None:
        NUM_PERM = args.num_perm
    elif args.num_per is not None:
        NUM_PERM = args.num_per
        log.warning(f"--num-per is a typo-alias for --num-perm; using NUM_PERM={NUM_PERM}")

    use_columns = [FALLBACK_COL] if args.question_only else args.columns

    files = _collect_parquet_files(args.input)
    log.info(f"Found {len(files)} parquet file(s)")

    t_total = time.perf_counter()
    for idx, fpath in enumerate(files, 1):
        out_path = _resolve_output_path(fpath, args.input, args.output)
        if args.skip_existing and os.path.exists(out_path):
            log.info(f"[{idx}/{len(files)}] SKIP (exists): {out_path}")
            continue
        log.info(f"[{idx}/{len(files)}] {fpath}  →  {out_path}")
        try:
            run_pipeline(
                fpath, out_path,
                columns=use_columns,
                combine=COMBINE_COLS,
                target_tasks_per_worker=args.tasks_per_worker,
                max_ram_per_worker_gb=args.max_ram_per_worker,
                min_task_size=args.min_task_size,
                max_task_size=args.max_task_size,
            )
        except Exception as exc:
            log.error(f"FAILED: {fpath}  —  {exc}", exc_info=True)

    log.info(f"All done in {(time.perf_counter()-t_total)/60:.1f} min")
