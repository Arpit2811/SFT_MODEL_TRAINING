from trl import SFTTrainer, SFTConfig
import torch
import torch.nn as nn
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
import pandas as pd
from datasets import Dataset
from datasets import load_dataset, concatenate_datasets
import inspect
import wandb
from datetime import timedelta
import torch.distributed as dist
import os

dist.init_process_group(
        backend='nccl',       # 'nccl' for GPUs, 'gloo' for CPU (or cross-platform)
        timeout=timedelta(days = 1))

result_dir="/fsxnew/arpit.kumble/main/results"

if os.environ["LOCAL_RANK"] == 0:
    run = wandb.init(
        project="Mix_Data_SFT",
        dir=result_dir, 
        name="Mix_Data_SFT_1",
    )

base_dir = "/fsxnew/arpit.kumble/main/"

ds_1 = load_dataset("parquet", data_files=f"{base_dir}/train/benchmark_incomplete_qa_deduped_clean_pass.parquet")["train"]
ds_2 = load_dataset("parquet", data_files=f"{base_dir}/train/benchmark_qa_0_3000000_0902260043_final_clean_clean_clean_clean_clean_pass.parquet")["train"]
ds_3 = load_dataset("parquet", data_files=f"{base_dir}/train/benchmark_qa_3000000_6000000_0802261457_2296199_clean_clean_clean_clean_clean_pass.parquet")["train"]
ds_4 = load_dataset("parquet", data_files=f"{base_dir}/train/benchmark_qa_6000000_9000000_1002262247_2994199_clean_clean_clean_clean_clean_pass.parquet")["train"]
ds_5 = load_dataset("parquet", data_files=f"{base_dir}/train/benchmark_qa_9000000_12000000_1102260257_31769_clean_clean_clean_clean_clean_pass.parquet")["train"]
ds_6 = load_dataset("parquet", data_files=f"{base_dir}/train/contextqa_ipqa_clean_pass.parquet")["train"]
ds_7 = load_dataset("parquet", data_files=f"{base_dir}/train/grammar_iqa_deduped_clean_clean_pass.parquet")["train"]
ds_8 = load_dataset("parquet", data_files=f"{base_dir}/train/hf_bank_faq_train_no_think_pass.parquet")["train"]
ds_9 = load_dataset("parquet", data_files=f"{base_dir}/train/hf_ca_train_no_think_pass.parquet")["train"]
ds_10 = load_dataset("parquet", data_files=f"{base_dir}/train/hf_ca_train_think_clean_pass.parquet")["train"]
ds_11 = load_dataset("parquet", data_files=f"{base_dir}/train/hf_fin_no_think_dedup_clean_clean_pass.parquet")["train"]
ds_12 = load_dataset("parquet", data_files=f"{base_dir}/train/hf_fin_think_dedup_clean_clean_pass.parquet")["train"]
ds_13 = load_dataset("parquet", data_files=f"{base_dir}/train/hf_math_1_train_no_think_dedup_clean_clean_pass.parquet")["train"]
ds_14 = load_dataset("parquet", data_files=f"{base_dir}/train/hf_math_1_train_think_dedup_clean_clean_pass.parquet")["train"]
ds_15 = load_dataset("parquet", data_files=f"{base_dir}/train/hf_math_train_no_think_pass.parquet")["train"]
ds_16 = load_dataset("parquet", data_files=f"{base_dir}/train/hf_math_train_think_deduped_clean_pass.parquet")["train"]
ds_17 = load_dataset("parquet", data_files=f"{base_dir}/train/hf_old_dataset_train_no_think_dedup_clean_pass.parquet")["train"]
ds_18 = load_dataset("parquet", data_files=f"{base_dir}/train/hf_tax_train_think_deduped_clean_pass.parquet")["train"]
ds_19 = load_dataset("parquet", data_files=f"{base_dir}/train/long_context_iqa_deduped_clean_clean_pass.parquet")["train"] 
ds_20 = load_dataset("parquet", data_files=f"{base_dir}/train/multistep_iqa_deduped_clean_clean_pass.parquet")["train"]
ds_21 = load_dataset("parquet", data_files=f"{base_dir}/train/multitask_qa_0402262040_1228999_clean_clean_clean_clean_clean_pass.parquet")["train"]
ds_22 = load_dataset("parquet", data_files=f"{base_dir}/train/nli_iqa_deduped_clean_clean_pass.parquet")["train"]
ds_23 = load_dataset("parquet", data_files=f"{base_dir}/train/python_qna_parquet_0006.parquet")["train"]
ds_24 = load_dataset("parquet", data_files=f"{base_dir}/train/sentence_expansion_deduped_clean_clean_pass_formatted.parquet")["train"]
ds_25 = load_dataset("parquet", data_files=f"{base_dir}/train/tally_qa_deduped_clean_pass.parquet")["train"]
ds_26 = load_dataset("parquet", data_files=f"{base_dir}/train/template_iqa_deduped_clean_clean_pass.parquet")["train"]

train_1 = concatenate_datasets([ds_1, ds_2, ds_3, ds_4, ds_5, ds_6, ds_7, ds_8, ds_9, ds_10, ds_11, ds_12, ds_13, ds_14, ds_15, ds_16, ds_17, ds_18, ds_19, ds_20, ds_21, ds_22, ds_23, ds_24, ds_25, ds_26]).shuffle(seed=42)
#print(train_2)

train_dataset = concatenate_datasets([train_1]).shuffle(seed=42)
print(train_dataset)


val_1 = load_dataset("parquet", data_files=f"{base_dir}/val/benchmark_incomplete_qa_deduped_clean_pass.parquet")["train"]
val_2 = load_dataset("parquet", data_files=f"{base_dir}/val/benchmark_qa_3000000_6000000_0802261457_2296199_clean_clean_clean_clean_clean_pass.parquet")["train"]
val_3 = load_dataset("parquet", data_files=f"{base_dir}/val/contextqa_ipqa_clean_pass.parquet")["train"]
val_4 = load_dataset("parquet", data_files=f"{base_dir}/val/grammar_iqa_deduped_clean_clean_pass.parquet")["train"]
val_5 = load_dataset("parquet", data_files=f"{base_dir}/val/hf_bank_faq_train_no_think_pass.parquet")["train"]
val_6 = load_dataset("parquet", data_files=f"{base_dir}/val/hf_ca_train_no_think_pass.parquet")["train"]
val_7 = load_dataset("parquet", data_files=f"{base_dir}/val/hf_ca_train_think_clean_pass.parquet")["train"]
val_8 = load_dataset("parquet", data_files=f"{base_dir}/val/hf_fin_no_think_dedup_clean_clean_pass.parquet")["train"]
val_9 = load_dataset("parquet", data_files=f"{base_dir}/val/hf_fin_think_dedup_clean_clean_pass.parquet")["train"]
val_10 = load_dataset("parquet", data_files=f"{base_dir}/val/hf_math_1_train_no_think_dedup_clean_clean_pass.parquet")["train"]
val_11 = load_dataset("parquet", data_files=f"{base_dir}/val/hf_math_1_train_think_dedup_clean_clean_pass.parquet")["train"]
val_12 = load_dataset("parquet", data_files=f"{base_dir}/val/hf_math_train_no_think_pass.parquet")["train"]
val_13 = load_dataset("parquet", data_files=f"{base_dir}/val/hf_math_train_think_deduped_clean_pass.parquet")["train"]
val_14 = load_dataset("parquet", data_files=f"{base_dir}/val/hf_old_dataset_train_no_think_dedup_clean_pass.parquet")["train"]
val_15 = load_dataset("parquet", data_files=f"{base_dir}/val/hf_tax_train_think_deduped_clean_pass.parquet")["train"]
val_16 = load_dataset("parquet", data_files=f"{base_dir}/val/long_context_iqa_deduped_clean_clean_pass.parquet")["train"]
val_17 = load_dataset("parquet", data_files=f"{base_dir}/val/multistep_iqa_deduped_clean_clean_pass.parquet")["train"]
val_18 = load_dataset("parquet", data_files=f"{base_dir}/val/multitask_qa_0402262040_1228999_clean_clean_clean_clean_clean_pass.parquet")["train"]
val_19 = load_dataset("parquet", data_files=f"{base_dir}/val/nli_iqa_deduped_clean_clean_pass.parquet")["train"]
val_20 = load_dataset("parquet", data_files=f"{base_dir}/val/python_qna_parquet_0006.parquet")["train"]
val_21 = load_dataset("parquet", data_files=f"{base_dir}/val/sentence_expansion_deduped_clean_clean_pass_formatted.parquet")["train"]
val_22 = load_dataset("parquet", data_files=f"{base_dir}/val/tally_qa_deduped_clean_pass.parquet")["train"]
val_23 = load_dataset("parquet", data_files=f"{base_dir}/val/template_iqa_deduped_clean_clean_pass.parquet")["train"]


eval_dsets = {
    "Benchmark_incomplete_qa": val_1,
    "Benchmarks": val_2,
    "Contextqa_ipqa": val_3,
    "Grammar_iqa": val_4,
    "hf_bank_faq_no_think": val_5,
    "hf_ca_no_think": val_6,
    "hf_ca_think": val_7,
    "hf_fin_no_think": val_8,
    "hf_fin_think": val_9,
    "hf_math_1_no_think": val_10,
    "hf_math_1_think": val_11,
    "hf_math_no_think": val_12,
    "hf_math_think": val_13,
    "hf_old_dataset_no_think": val_14,
    "hf_tax_think": val_15,
    "long_context_iqa": val_16,
    "multistep_iqa": val_17,
    "multitask_qa": val_18,
    "nli_iqa": val_19,
    "python_qna": val_20,
    "sentence_expansion": val_21,
    "tally_qa": val_22,
    "template_iqa": val_23,
}


model = AutoModelForCausalLM.from_pretrained("/home/aamod.thakur/Model/All_Models/Param_1B", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/fsxnew/arpit.kumble/param2_tokenizer", max_length=2048)

training_args = SFTConfig(

    ### Keep as we don't want group_id to be removed
    #remove_unused_columns=False,
    assistant_only_loss=True,
    
    #dataset_text_field="text",
    dataset_num_proc=120,

    output_dir=result_dir,
    overwrite_output_dir=True,

    num_train_epochs=2,       ### Keep as it is
    logging_steps=100,
    learning_rate=5e-5,

    do_train=True,
    do_eval=True,

    per_device_train_batch_size=6,
    per_device_eval_batch_size =6,

    gradient_accumulation_steps = 15,

    save_steps=2000,
    save_total_limit=-1,

    #lr_scheduler_type = "cosine",
    #warmup_steps = 10,
    lr_scheduler_type = "constant",

    max_grad_norm = 0.5,

    optim='adamw_torch',
    adam_epsilon = 5e-5,
    adam_beta2 = 0.98,

    eval_strategy="steps",
    eval_steps=2000,
    packing=False,

    use_cpu=False,
    ddp_find_unused_parameters=False,

    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dsets,
    
    args=training_args,
    processing_class=tokenizer,
)

trainer.train()
#trainer.train(resume_from_checkpoint="/projects/data/aamod/Results/FinSFT_1/checkpoint-15000")
