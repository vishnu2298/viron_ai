"""
Train one LoRA skill adapter on top of a small base model (QLoRA, 4-bit).
Runs on a single 16-24 GB GPU. Output: a ~10-100 MB adapter, not a new model.

IMPORTANT: an adapter is tied to its base weights. Train separately for each tier
(tiny base and core base) if you want the same skill on both.

Usage:
  python train_lora.py --base <hf-model-id> --data data/support.jsonl --out adapters/support-core
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.pad_token or tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=args.data, split="train").train_test_split(test_size=0.05, seed=42)

    def tokenize(ex):
        text = tok.apply_chat_template(ex["messages"], tokenize=False)
        return tok(text, truncation=True, max_length=args.max_len)

    ds = ds.map(tokenize, remove_columns=["messages"])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to="none",
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"adapter saved -> {args.out}")


if __name__ == "__main__":
    main()
