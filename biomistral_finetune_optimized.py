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
import gc
from tqdm import tqdm

# Set device and memory management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear cache
torch.cuda.empty_cache() if torch.cuda.is_available() else None

def load_and_prepare_data():
    """Load and prepare the medical dataset"""
    print("Loading full dataset...")
    df = pd.read_csv("DiseaseAndSymptoms_with_Criticality.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Check for missing values
    print(f"Missing values in Criticality: {df['Criticality'].isna().sum()}")
    
    # Remove rows with missing criticality
    df = df.dropna(subset=['Criticality'])
    print(f"Dataset shape after removing missing criticality: {df.shape}")
    
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
    prompts = []
    for text, crit in tqdm(zip(texts, criticalities), total=len(texts), desc="Creating prompts"):
        prompts.append(create_instruction_prompt(text, crit))
    
    return prompts

def setup_model_and_tokenizer():
    """Setup BioMistral model and tokenizer with optimization"""
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
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA with more conservative settings
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Reduced rank for memory efficiency
        lora_alpha=16,  # Reduced alpha
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Only attention layers
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def create_datasets(prompts, tokenizer):
    """Create training and validation datasets"""
    # Split data
    train_prompts, val_prompts = train_test_split(prompts, test_size=0.1, random_state=42)
    print(f"Training samples: {len(train_prompts)}")
    print(f"Validation samples: {len(val_prompts)}")

    # Tokenize data in batches to avoid memory issues
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=256,  # Reduced max length for memory efficiency
            return_tensors="pt"
        )

    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_prompts})
    val_dataset = Dataset.from_dict({"text": val_prompts})

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=100)
    val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=100)
    
    return train_dataset, val_dataset

def train_model(model, tokenizer, train_dataset, val_dataset):
    """Train the model with optimized settings"""
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Training arguments with memory optimization
    training_args = TrainingArguments(
        output_dir="./biomistral-medical-criticality",
        per_device_train_batch_size=1,  # Reduced batch size
        per_device_eval_batch_size=1,   # Reduced batch size
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size
        num_train_epochs=2,  # Reduced epochs
        learning_rate=1e-4,  # Reduced learning rate
        fp16=True,
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=50,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # Reduce memory usage
        gradient_checkpointing=True,  # Enable gradient checkpointing
        max_grad_norm=1.0,  # Gradient clipping
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
    try:
        trainer.train()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
        return False
    
    return True

def save_model(model, tokenizer, success):
    """Save the trained model"""
    if success:
        print("Saving model...")
        try:
            # Save the full model
            trainer.save_model()
            tokenizer.save_pretrained("./biomistral-medical-criticality")
            
            # Save LoRA adapters separately
            model.save_pretrained("./biomistral-medical-criticality-lora")
            
            print("✅ Model saved successfully!")
            print("📁 Full model: ./biomistral-medical-criticality/")
            print("📁 LoRA adapters: ./biomistral-medical-criticality-lora/")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    else:
        print("❌ Model training failed, not saving.")

def main():
    """Main training pipeline"""
    try:
        # Load and prepare data
        prompts = load_and_prepare_data()
        
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer()
        
        # Create datasets
        train_dataset, val_dataset = create_datasets(prompts, tokenizer)
        
        # Clear memory
        del prompts
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Train model
        success = train_model(model, tokenizer, train_dataset, val_dataset)
        
        # Save model
        save_model(model, tokenizer, success)
        
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
