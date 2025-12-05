import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the full dataset
print("Loading full dataset...")
df = pd.read_csv("DiseaseAndSymptoms_with_Criticality.csv")
print(f"Dataset shape: {df.shape}")

def combine_symptoms_and_disease(row):
    """Combine symptoms and disease into a single text"""
    symptoms = [str(row[col]) for col in row.index if "Symptom" in col and pd.notna(row[col])]
    disease = str(row["Disease"]) if pd.notna(row["Disease"]) else ""
    
    symptoms_text = " ".join(symptoms)
    return f"Disease: {disease}\nSymptoms: {symptoms_text}"

def create_instruction_prompt(text, criticality):
    """Create instruction-following prompt for BioMistral"""
    return f"""<s>[INST] You are a medical AI assistant. Analyze the following medical information and determine the criticality level.

Medical Information:
{text}

Please classify the criticality as one of: Mild, Moderate, or Critical.

Criticality: [/INST] {criticality}</s>"""

# Prepare data
print("Preparing training data...")
texts = df.apply(combine_symptoms_and_disease, axis=1).tolist()
criticalities = df["Criticality"].tolist()

# Create instruction prompts
prompts = [create_instruction_prompt(text, crit) for text, crit in zip(texts, criticalities)]

# Split data
train_prompts, val_prompts = train_test_split(prompts, test_size=0.1, random_state=42)
print(f"Training samples: {len(train_prompts)}")
print(f"Validation samples: {len(val_prompts)}")

# Load BioMistral model and tokenizer
model_name = "microsoft/BioMistral-7B"
print(f"Loading {model_name}...")

# Configure quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize data
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

# Create datasets
train_dataset = Dataset.from_dict({"text": train_prompts})
val_dataset = Dataset.from_dict({"text": val_prompts})

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./biomistral-medical-criticality",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_steps=100,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    report_to="none",
    remove_unused_columns=False
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the model
print("Saving model...")
trainer.save_model()
tokenizer.save_pretrained("./biomistral-medical-criticality")

# Save LoRA adapters separately
model.save_pretrained("./biomistral-medical-criticality-lora")

print("Training completed! Model saved to ./biomistral-medical-criticality/")
print("LoRA adapters saved to ./biomistral-medical-criticality-lora/")
