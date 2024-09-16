import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# Model and dataset configuration
model_name = "Llama-2-7b-chat-hf"
dataset_name = " "  # Replace with your dataset name
new_model_name = "Llama-2-7b-chat-finetune"

# LoRA configuration
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

# bitsandbytes configuration
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# Training configuration
output_dir = " "  # Replace with your desired output directory
num_epochs = 5
fp16 = True  # Enable mixed precision to utilize GPU more efficiently
bf16 = False  # Set to True if using bfloat16 precision and supported by GPU
batch_size = 8  # Adjust this according to your available GPU memory
grad_accum_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.1
learning_rate = 1e-4
weight_decay = 1e-5
optimizer = "paged_adamw_32bit"  # Ensure this optimizer is supported by your setup
lr_scheduler = "cosine"
group_by_length = True
max_seq_length = None  # Adjust if needed for your task
packing = False

device = "cuda" if torch.cuda.is_available() else "cpu"

# Helper function to configure bitsandbytes
def configure_bnb():
    return BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=use_nested_quant,
    )

# Load dataset
dataset = load_dataset(dataset_name, split="train") # prepare your own dataset


# Load base model and tokenizer
bnb_config = configure_bnb()

# Load the model and move it to the appropriate device
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=bnb_config, 
    device_map="auto"  # Automatically map to available GPUs
)
model.to(device)

# Disable caching for training
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load and configure tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix overflow issue with fp16

# Configure LoRA (Low-Rank Adaptation)
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum_steps,
    optim=optimizer,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",  # Make sure the dataset contains a 'text' field
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
    packing=packing,
)

# Start training and save the model
trainer.train()
trainer.model.save_pretrained(new_model_name)
