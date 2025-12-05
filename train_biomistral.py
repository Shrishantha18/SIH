#!/usr/bin/env python3
"""
BioMistral-7B Medical Criticality Classification Training Script

This script fine-tunes BioMistral-7B on your medical dataset for criticality classification.
It uses LoRA (Low-Rank Adaptation) for efficient fine-tuning.

Usage:
    python train_biomistral.py

Requirements:
    - CUDA-compatible GPU with at least 8GB VRAM
    - Install requirements: pip install -r requirements.txt
"""

import os
import sys
import torch
import pandas as pd
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
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This script requires a GPU.")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check if dataset exists
    if not os.path.exists("DiseaseAndSymptoms_with_Criticality.csv"):
        print("❌ Dataset file not found: DiseaseAndSymptoms_with_Criticality.csv")
        return False
    
    print("✅ Dataset file found")
    return True

def load_data():
    """Load and prepare the medical dataset"""
    print("\n📊 Loading dataset...")
    
    df = pd.read_csv("DiseaseAndSymptoms_with_Criticality.csv")
    print(f"📈 Dataset shape: {df.shape}")
    
    # Check criticality distribution
    criticality_counts = df['Criticality'].value_counts()
    print(f"📊 Criticality distribution:\n{criticality_counts}")
    
    # Remove missing values
    df = df.dropna(subset=['Criticality'])
    print(f"📈 After removing missing values: {df.shape}")
    
    def create_medical_text(row):
        """Create medical text from symptoms and disease"""
        symptoms = [str(row[col]) for col in row.index if "Symptom" in col and pd.notna(row[col])]
        disease = str(row["Disease"]) if pd.notna(row["Disease"]) else ""
        symptoms_text = " ".join(symptoms)
        return f"Disease: {disease}\nSymptoms: {symptoms_text}"

    def create_prompt(text, criticality):
        """Create instruction prompt for BioMistral"""
        return f"""<s>[INST] You are a medical AI assistant. Analyze the following medical information and determine the criticality level.

Medical Information:
{text}

Please classify the criticality as one of: Mild, Moderate, or Critical.

Criticality: [/INST] {criticality}</s>"""

    # Create prompts
    print("🔄 Creating training prompts...")
    texts = df.apply(create_medical_text, axis=1).tolist()
    criticalities = df["Criticality"].tolist()
    
    prompts = []
    for text, crit in tqdm(zip(texts, criticalities), total=len(texts), desc="Creating prompts"):
        prompts.append(create_prompt(text, crit))
    
    return prompts

def setup_model():
    """Setup BioMistral model with LoRA"""
    print("\n🤖 Setting up BioMistral model...")
    
    model_name = "microsoft/BioMistral-7B"
    
    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    print("🧠 Loading BioMistral model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def create_datasets(prompts, tokenizer):
    """Create training and validation datasets"""
    print("\n📚 Creating datasets...")
    
    # Split data
    train_prompts, val_prompts = train_test_split(prompts, test_size=0.1, random_state=42)
    print(f"📊 Training samples: {len(train_prompts)}")
    print(f"📊 Validation samples: {len(val_prompts)}")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
    
    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_prompts})
    val_dataset = Dataset.from_dict({"text": val_prompts})
    
    print("🔄 Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=50)
    val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=50)
    
    return train_dataset, val_dataset

def train_model(model, tokenizer, train_dataset, val_dataset):
    """Train the model"""
    print("\n🚀 Starting training...")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./biomistral-medical-criticality",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=1e-4,
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
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
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
    
    # Train
    try:
        trainer.train()
        print("✅ Training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def save_model(model, tokenizer, success):
    """Save the trained model"""
    if success:
        print("\n💾 Saving model...")
        try:
            # Save full model
            model.save_pretrained("./biomistral-medical-criticality")
            tokenizer.save_pretrained("./biomistral-medical-criticality")
            
            # Save LoRA adapters
            model.save_pretrained("./biomistral-medical-criticality-lora")
            
            print("✅ Model saved successfully!")
            print("📁 Full model: ./biomistral-medical-criticality/")
            print("📁 LoRA adapters: ./biomistral-medical-criticality-lora/")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    else:
        print("❌ Training failed, not saving model.")

def main():
    """Main training pipeline"""
    print("🏥 BioMistral Medical Criticality Training")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    try:
        # Load data
        prompts = load_data()
        
        # Setup model
        model, tokenizer = setup_model()
        
        # Create datasets
        train_dataset, val_dataset = create_datasets(prompts, tokenizer)
        
        # Clear memory
        del prompts
        gc.collect()
        torch.cuda.empty_cache()
        
        # Train model
        success = train_model(model, tokenizer, train_dataset, val_dataset)
        
        # Save model
        save_model(model, tokenizer, success)
        
        if success:
            print("\n🎉 Training completed successfully!")
            print("You can now use the model with biomistral_inference.py")
        else:
            print("\n💥 Training failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
