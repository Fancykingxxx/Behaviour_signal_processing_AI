import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = os.environ.get("DATA_DIR", "./training_data_set")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs/qwen2p5_7b_lora")

train_file = os.path.join(DATA_DIR, "train.jsonl")
val_file = os.path.join(DATA_DIR, "val.jsonl")

def format_example(example):
    text = (
        f"Instruction: {example['instruction']}\n\n"
        f"Input: {example['input']}\n\n"
        f"Output: {example['output']}"
    )
    return {"text": text}

dataset = load_dataset(
    "json",
    data_files={"train": train_file, "validation": val_file}
)

dataset = dataset.map(format_example)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    bf16=True,
    max_seq_length=1024,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    packing=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    peft_config=peft_config,
    dataset_text_field="text",
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Training complete. Model saved to: {OUTPUT_DIR}")