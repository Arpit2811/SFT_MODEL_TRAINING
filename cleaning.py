"""
Optimized Data Cleaning Pipeline — 50M+ rows (HARDENED & DIAGNOSTIC VERSION)
================================================================
- Python-side merging (no binary_join_element_wise kernel issues)
- Strict long-string row dropping
- Per-worker logging + timing to detect hangs
- Safer parallelism defaults
- Character-level repetition detection
- Token density check (catches giant single-token blobs)
- Total merged row byte check before processing
- Abnormal token count detection (z-score) + targeted repetition check
"""

import os
import io
import glob
import logging
import argparse
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID %(process)d] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────── DROP LONG STRINGS ──────────────────────

def _drop_rows_with_long_strings(
    tbl: pa.Table,
    columns: list[str],
    max_length_bytes: int = 5_000_000,
) -> tuple[pa.Table, int]:
    start = time.time()
    long_mask = pa.array([False] * len(tbl), type=pa.bool_())

    for col_name in columns:
        col = tbl.column(col_name)
        lengths = pc.utf8_length(col)
        is_long = pc.greater(lengths, pa.scalar(max_length_bytes, pa.int64()))
        long_mask = pc.or_(long_mask, is_long)

    num_long = int(pc.sum(long_mask).as_py() or 0)
    if num_long > 0:
        tbl = tbl.filter(pc.invert(long_mask))
        log.info(f"Dropped {num_long:,} rows > {max_length_bytes:,} bytes in {time.time()-start:.1f}s")

    return tbl, num_long


# ──────────────────────── REPETITION RATIO ──────────────────────

def _repetition_scores(texts: list[str], n: int) -> np.ndarray:
    scores = np.zeros(len(texts), dtype=np.float32)
    for idx, text in enumerate(texts):
        if not text:
            continue
        words = text.split()
        total = len(words) - n + 1
        if total <= 0:
            continue
        ngrams = [tuple(words[i:i + n]) for i in range(total)]
        counts = Counter(ngrams)
        repeated = total - len(counts)
        scores[idx] = repeated / total
    return scores


# ──────────── CHARACTER-LEVEL REPETITION CHECK (FAST REGEX) ──────

def _is_char_repetitive(
    text: str,
    min_run: int = 200,
    threshold: float = 0.3,
) -> bool:
    """
    Fast O(n) check using regex for runs of repeated single chars
    or 2-char patterns that cover >= threshold of the total text.
    Catches embedded garbage like u2011 u2011 u2011... or xa0...xa0...
    even when surrounded by legitimate prose.
    """
    if not text:
        return False
    text_len = len(text)

    # Single char runs: aaaaaa...
    for m in re.finditer(r'(.)\1{' + str(min_run - 1) + r',}', text):
        if len(m.group()) / text_len >= threshold:
            return True

    # 2-char pattern runs: ababab...
    for m in re.finditer(r'(..)\1{' + str(min_run // 2 - 1) + r',}', text):
        if len(m.group()) / text_len >= threshold:
            return True

    return False


# ──────────── TOKEN DENSITY CHECK ────────────────────────────────

def _is_token_sparse(
    text: str,
    max_avg_token_len: int = 50,
) -> bool:
    """
    Returns True if the average token (whitespace-split word) is
    suspiciously long. A very high average means the text is a blob
    of garbage rather than natural language.
    """
    if not text:
        return False
    tokens = text.split()
    if not tokens:
        return False
    avg_len = len(text) / len(tokens)
    return avg_len > max_avg_token_len


# ──────────── MERGED ROW BYTE CHECK ──────────────────────────────

def _drop_rows_with_long_merged(
    tbl: pa.Table,
    merged_texts: list[str],
    max_merged_bytes: int,
) -> tuple[pa.Table, list[str], int]:
    """
    Drop rows where the total merged text length across all columns
    exceeds max_merged_bytes.
    """
    keep = [len(t) <= max_merged_bytes for t in merged_texts]
    removed = sum(1 for k in keep if not k)
    if removed > 0:
        tbl = tbl.filter(pa.array(keep, pa.bool_()))
        merged_texts = [t for t, k in zip(merged_texts, keep) if k]
        log.info(f"Dropped {removed:,} rows with merged text > {max_merged_bytes:,} bytes")
    return tbl, merged_texts, removed


# ──────────── ABNORMAL TOKEN COUNT DETECTION ─────────────────────

def _find_abnormal_token_count_rows(
    token_counts: list[int],
    z_score_threshold: float = 3.0,
) -> list[bool]:
    """
    Returns a boolean list where True = abnormally high token count.
    Uses z-score: anything more than z_score_threshold std devs above
    the mean is considered abnormal.
    """
    arr = np.array(token_counts, dtype=np.float32)
    mean = arr.mean()
    std = arr.std()
    if std == 0:
        return [False] * len(token_counts)
    z_scores = (arr - mean) / std
    return [float(z) > z_score_threshold for z in z_scores]


def _drop_abnormal_token_repetitive_rows(
    tbl: pa.Table,
    merged_texts: list[str],
    token_count_columns: list[str],
    z_score_threshold: float = 3.0,
) -> tuple[pa.Table, list[str], int]:
    """
    For each token count column passed:
      1. Find rows with abnormally high token count (z-score)
      2. Run _is_char_repetitive ONLY on those rows
      3. Drop rows that are both abnormal AND repetitive

    This avoids running expensive regex on every row —
    only statistical outliers get the deep check.
    """
    if not token_count_columns:
        return tbl, merged_texts, 0

    n = len(merged_texts)
    is_garbage = [False] * n

    for col_name in token_count_columns:
        token_counts = tbl.column(col_name).to_pylist()
        abnormal_mask = _find_abnormal_token_count_rows(token_counts, z_score_threshold)

        checked = 0
        for i, (text, is_abnormal) in enumerate(zip(merged_texts, abnormal_mask)):
            if not is_garbage[i] and is_abnormal:
                checked += 1
                if _is_char_repetitive(text):
                    is_garbage[i] = True

        log.info(f"Column '{col_name}': checked {checked:,} abnormal rows out of {n:,} total")

    garbage_removed = sum(is_garbage)
    if garbage_removed > 0:
        keep = [not g for g in is_garbage]
        tbl = tbl.filter(pa.array(keep, pa.bool_()))
        merged_texts = [t for t, k in zip(merged_texts, keep) if k]
        log.info(f"Dropped {garbage_removed:,} rows with abnormal token count + repetitive content")

    return tbl, merged_texts, garbage_removed


# ──────────────────── IPC TRANSPORT ──────────────────────

def _serialize_table(tbl: pa.Table) -> bytes:
    sink = io.BytesIO()
    with ipc.new_stream(sink, tbl.schema) as w:
        for b in tbl.to_batches():
            w.write_batch(b)
    return sink.getvalue()


def _deserialize_table(data: bytes) -> pa.Table:
    return ipc.open_stream(io.BytesIO(data)).read_all()


# ──────────── WORKER ─────────

def _process_row_group(
    file_path: str,
    rg_index: int,
    columns: list[str],
    token_count_columns: list[str],
    ngram_n: int,
    repeat_threshold: float,
    max_string_bytes: int,
    z_score_threshold: float,
) -> tuple[bytes | None, dict]:
    pid = os.getpid()
    log.info(f"Starting row group {rg_index} of {file_path}")

    empty = {
        "total": 0,
        "long_removed": 0,
        "long_merged_removed": 0,
        "chinese_removed": 0,
        "char_rep_removed": 0,
        "token_sparse_removed": 0,
        "abnormal_token_rep_removed": 0,
        "repetition_removed": 0,
        "final": 0,
    }

    try:
        start = time.time()
        pf = pq.ParquetFile(file_path)

        # Read text columns + token count columns together in one pass
        all_cols_to_read = list(set(columns + token_count_columns))
        tbl = pf.read_row_group(rg_index, columns=all_cols_to_read)

        # Cast string_view -> string if needed
        new_cols = []
        for col in tbl.columns:
            if str(col.type) == "string_view":
                new_cols.append(col.cast(pa.string()))
            else:
                new_cols.append(col)
        tbl = pa.table(new_cols, names=tbl.schema.names)
        log.info(f"Read row group {rg_index} in {time.time()-start:.1f}s, rows={len(tbl)}")

        # Null filter (only on text columns)
        start = time.time()
        valid_masks = [pc.is_valid(tbl.column(c)) for c in columns]
        valid = valid_masks[0]
        for m in valid_masks[1:]:
            valid = pc.and_(valid, m)
        tbl = tbl.filter(valid)
        log.info(f"After null filter: {len(tbl)} rows ({time.time()-start:.1f}s)")

        total = len(tbl)
        if total == 0:
            return None, empty

        # Per-column long strings drop
        tbl, num_long = _drop_rows_with_long_strings(tbl, columns, max_string_bytes)

        if len(tbl) == 0:
            return None, {**empty, "total": total, "long_removed": num_long}

        # Safe Python merge of text columns only
        start = time.time()
        col_lists = [tbl.column(c).to_pylist() for c in columns]
        log.info(f"to_pylist completed in {time.time()-start:.1f}s")

        start = time.time()
        merged_texts = [
            " ".join(str(v) if v is not None else "" for v in row_vals)
            for row_vals in zip(*col_lists)
        ]
        log.info(f"Merged {len(merged_texts)} texts in {time.time()-start:.1f}s")

        # Total merged row byte check
        tbl, merged_texts, num_long_merged = _drop_rows_with_long_merged(
            tbl, merged_texts, max_string_bytes
        )
        if len(tbl) == 0:
            return None, {**empty, "total": total, "long_removed": num_long,
                          "long_merged_removed": num_long_merged}

        # Chinese filter
        start = time.time()
        is_chinese_list = [bool(re.search(r'[一-鿿]', text)) for text in merged_texts]
        chinese_removed = sum(is_chinese_list)
        log.info(f"Chinese scan done in {time.time()-start:.1f}s, removed {chinese_removed}")

        keep_chinese = [not x for x in is_chinese_list]
        tbl = tbl.filter(pa.array(keep_chinese, pa.bool_()))
        merged_texts = [t for t, k in zip(merged_texts, keep_chinese) if k]

        if len(tbl) == 0:
            return None, {**empty, "total": total, "long_removed": num_long,
                          "long_merged_removed": num_long_merged,
                          "chinese_removed": chinese_removed}

        # Character-level repetition filter
        start = time.time()
        is_char_rep_list = [_is_char_repetitive(text) for text in merged_texts]
        char_rep_removed = sum(is_char_rep_list)
        keep_char_rep = [not x for x in is_char_rep_list]
        tbl = tbl.filter(pa.array(keep_char_rep, pa.bool_()))
        merged_texts = [t for t, k in zip(merged_texts, keep_char_rep) if k]
        log.info(f"Char-level repetition filter done in {time.time()-start:.1f}s, removed {char_rep_removed}")

        if len(tbl) == 0:
            return None, {**empty, "total": total, "long_removed": num_long,
                          "long_merged_removed": num_long_merged,
                          "chinese_removed": chinese_removed,
                          "char_rep_removed": char_rep_removed}

        # Token density (sparse token) filter
        start = time.time()
        is_sparse_list = [_is_token_sparse(text) for text in merged_texts]
        token_sparse_removed = sum(is_sparse_list)
        keep_sparse = [not x for x in is_sparse_list]
        tbl = tbl.filter(pa.array(keep_sparse, pa.bool_()))
        merged_texts = [t for t, k in zip(merged_texts, keep_sparse) if k]
        log.info(f"Token density filter done in {time.time()-start:.1f}s, removed {token_sparse_removed}")

        if len(tbl) == 0:
            return None, {**empty, "total": total, "long_removed": num_long,
                          "long_merged_removed": num_long_merged,
                          "chinese_removed": chinese_removed,
                          "char_rep_removed": char_rep_removed,
                          "token_sparse_removed": token_sparse_removed}

        # Abnormal token count + targeted char repetition filter
        # Only rows flagged as statistical outliers get the deep regex check
        start = time.time()
        tbl, merged_texts, abnormal_token_rep_removed = _drop_abnormal_token_repetitive_rows(
            tbl, merged_texts, token_count_columns, z_score_threshold
        )
        log.info(
            f"Abnormal token+rep filter done in {time.time()-start:.1f}s, "
            f"removed {abnormal_token_rep_removed}"
        )

        if len(tbl) == 0:
            return None, {**empty, "total": total, "long_removed": num_long,
                          "long_merged_removed": num_long_merged,
                          "chinese_removed": chinese_removed,
                          "char_rep_removed": char_rep_removed,
                          "token_sparse_removed": token_sparse_removed,
                          "abnormal_token_rep_removed": abnormal_token_rep_removed}

        # Word-level n-gram repetition filter
        start = time.time()
        rep_scores = _repetition_scores(merged_texts, ngram_n)
        rep_removed = int((rep_scores > repeat_threshold).sum())
        keep_rep = rep_scores <= repeat_threshold
        tbl_clean = tbl.filter(pa.array(keep_rep, pa.bool_()))
        log.info(f"Repetition filter done in {time.time()-start:.1f}s, removed {rep_removed}")

        stats = {
            "total": total,
            "long_removed": num_long,
            "long_merged_removed": num_long_merged,
            "chinese_removed": chinese_removed,
            "char_rep_removed": char_rep_removed,
            "token_sparse_removed": token_sparse_removed,
            "abnormal_token_rep_removed": abnormal_token_rep_removed,
            "repetition_removed": rep_removed,
            "final": len(tbl_clean),
        }

        return (_serialize_table(tbl_clean) if len(tbl_clean) > 0 else None), stats

    except Exception as exc:
        log.exception(f"Worker {pid} failed in row group {rg_index}: {exc}")
        return None, {**empty, "error": str(exc)}


# ──────────────────────────── CLI ───────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Robust Parquet cleaner with diagnostics")
    p.add_argument("--input", "-i", required=True,
                   help="Input directory containing .parquet files")
    p.add_argument("--output", "-o", required=True,
                   help="Output directory for cleaned .parquet files")
    p.add_argument("--columns", "-c", nargs="+", required=True,
                   help="Text columns to clean (2-4 columns)")
    p.add_argument("--token-count-columns", "-tc", nargs="+", default=[],
                   help="Token count columns for abnormal detection. "
                        "Pass one or more column names that already exist "
                        "in the parquet, e.g.: -tc question_tokens answer_tokens. "
                        "Rows with z-score > threshold AND repetitive content "
                        "will be dropped.")
    p.add_argument("--workers", "-w", type=int, default=16,
                   help="Number of parallel workers (default: 16)")
    p.add_argument("--threshold", "-t", type=float, default=0.5,
                   help="N-gram repetition threshold (default: 0.5)")
    p.add_argument("--ngram", "-n", type=int, default=4,
                   help="N-gram size for repetition scoring (default: 4)")
    p.add_argument("--compression", default="snappy",
                   choices=["snappy", "gzip", "brotli", "zstd", "none"],
                   help="Output compression codec (default: snappy)")
    p.add_argument("--row-group-size", type=int, default=200_000,
                   help="Output row group size (default: 200000)")
    p.add_argument("--max-string-bytes", type=int, default=5_000_000,
                   help="Drop rows where any text cell exceeds this size in bytes "
                        "(default: 5MB)")
    p.add_argument("--z-score-threshold", type=float, default=3.0,
                   help="Z-score threshold for abnormal token count detection. "
                        "Rows exceeding this many std devs above the mean token "
                        "count will be checked for repetition (default: 3.0)")
    args = p.parse_args()
    if not (2 <= len(args.columns) <= 4):
        p.error("--columns requires 2-4 column names")
    return args


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    workers = args.workers

    log.info(
        f"Files: {len(files)} | Workers: {workers} | "
        f"Max string bytes: {args.max_string_bytes:,} | "
        f"Token count columns: {args.token_count_columns or 'none'} | "
        f"Z-score threshold: {args.z_score_threshold}"
    )

    if not files:
        log.warning("No .parquet files found in input directory.")
        return

    file_infos = []
    total_rows = 0
    total_tasks = 0
    errors = []

    for fp in files:
        name = os.path.basename(fp)
        out = os.path.join(args.output, name.replace(".parquet", "_clean.parquet"))
        try:
            pf = pq.ParquetFile(fp)
            schema_names = set(pf.schema_arrow.names)
            nr = pf.metadata.num_rows
            nrg = pf.metadata.num_row_groups
            total_rows += nr
            total_tasks += nrg

            missing_text = set(args.columns) - schema_names
            if missing_text:
                log.error(f"Skipping {name} — missing text columns: {missing_text}")
                errors.append(name)
                continue

            missing_token = set(args.token_count_columns) - schema_names
            if missing_token:
                log.error(f"Skipping {name} — missing token count columns: {missing_token}")
                errors.append(name)
                continue

            file_infos.append((fp, out, name, nrg))
        except Exception as e:
            log.error(f"{name} metadata error: {e}")
            errors.append(name)

    if not file_infos:
        log.error("No valid files to process.")
        return

    grand = {
        "total": 0,
        "long_removed": 0,
        "long_merged_removed": 0,
        "chinese_removed": 0,
        "char_rep_removed": 0,
        "token_sparse_removed": 0,
        "abnormal_token_rep_removed": 0,
        "repetition_removed": 0,
        "final": 0,
    }
    file_stats = defaultdict(lambda: {k: 0 for k in grand})

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for fp, out, name, nrg in file_infos:
            for rg_idx in range(nrg):
                fut = pool.submit(
                    _process_row_group,
                    fp, rg_idx,
                    args.columns,
                    args.token_count_columns,
                    args.ngram,
                    args.threshold,
                    args.max_string_bytes,
                    args.z_score_threshold,
                )
                futures[fut] = (out, name, rg_idx)

        writers = {}
        completed = 0

        pbar = tqdm(total=total_rows, desc="Cleaning", unit="row",
                    unit_scale=True, dynamic_ncols=True, colour="green")

        with pbar:
            for fut in as_completed(futures):
                out_path, fname, rg_idx = futures[fut]
                tbl_bytes, stats = fut.result()

                if "error" in stats:
                    log.warning(f"Row group {rg_idx} error in {fname}: {stats['error']}")

                for k in grand:
                    grand[k] += stats.get(k, 0)
                for k in file_stats[out_path]:
                    file_stats[out_path][k] += stats.get(k, 0)

                if tbl_bytes is not None:
                    clean_tbl = _deserialize_table(tbl_bytes)
                    if out_path not in writers:
                        writers[out_path] = pq.ParquetWriter(
                            out_path, clean_tbl.schema,
                            compression=args.compression)
                    writers[out_path].write_table(
                        clean_tbl, row_group_size=args.row_group_size)

                completed += 1
                pbar.update(stats.get("total", 0))
                pbar.set_postfix(
                    tasks=f"{completed}/{total_tasks}",
                    kept=f"{grand['final']:,}",
                    long=f"{grand['long_removed']:,}",
                    lmrg=f"{grand['long_merged_removed']:,}",
                    zh=f"{grand['chinese_removed']:,}",
                    chrep=f"{grand['char_rep_removed']:,}",
                    sparse=f"{grand['token_sparse_removed']:,}",
                    abnrep=f"{grand['abnormal_token_rep_removed']:,}",
                    rep=f"{grand['repetition_removed']:,}",
                    retain=f"{grand['final']/grand['total']*100:.1f}%"
                    if grand["total"] else "—",
                )

    for w in writers.values():
        w.close()

    # Final per-file report
    for out_p, st in file_stats.items():
        inp_name = os.path.basename(out_p).replace("_clean.parquet", ".parquet")
        retention = st["final"] / st["total"] * 100 if st["total"] else 0
        log.info(
            "✅ %s | total=%d kept=%d long=%d long_merged=%d zh=%d "
            "char_rep=%d sparse=%d abnrep=%d rep=%d retain=%.1f%%",
            inp_name, st["total"], st["final"],
            st["long_removed"], st["long_merged_removed"],
            st["chinese_removed"], st["char_rep_removed"],
            st["token_sparse_removed"], st["abnormal_token_rep_removed"],
            st["repetition_removed"], retention
        )

    g_ret = grand["final"] / grand["total"] * 100 if grand["total"] else 0
    print("\n" + "=" * 80)
    print("GRAND TOTAL")
    print(f"  Rows read                    : {grand['total']:,}")
    print(f"  Long (per-col) removed       : {grand['long_removed']:,}")
    print(f"  Long (merged) removed        : {grand['long_merged_removed']:,}")
    print(f"  Chinese removed              : {grand['chinese_removed']:,}")
    print(f"  Char-repetition removed      : {grand['char_rep_removed']:,}")
    print(f"  Token-sparse removed         : {grand['token_sparse_removed']:,}")
    print(f"  Abnormal token+rep removed   : {grand['abnormal_token_rep_removed']:,}")
    print(f"  Word-rep removed             : {grand['repetition_removed']:,}")
    print(f"  Final clean rows             : {grand['final']:,}")
    print(f"  Retention                    : {g_ret:.1f}%")
    if errors:
        print(f"  Files with errors            : {len(errors)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
