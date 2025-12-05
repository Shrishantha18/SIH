# BioMistral-7B Medical Criticality Training

This guide explains how to train BioMistral-7B on your medical dataset for criticality classification.

## Overview

We've upgraded from the small `Zabihin/Symptom_to_Diagnosis` model to BioMistral-7B, a large language model specifically designed for biomedical applications. This provides much better performance for medical text understanding and classification.

## Files Created

- `train_biomistral.py` - Main training script (recommended)
- `biomistral_finetune_optimized.py` - Advanced training script with memory optimization
- `biomistral_inference.py` - Inference script for the trained model
- `requirements.txt` - Required Python packages

## Prerequisites

1. **GPU Requirements**: You need a CUDA-compatible GPU with at least 8GB VRAM
2. **Python Environment**: Python 3.8+ with pip
3. **Dataset**: Your `DiseaseAndSymptoms_with_Criticality.csv` file

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Verify CUDA is available:
```python
import torch
print(torch.cuda.is_available())
```

## Training the Model

### Option 1: Simple Training (Recommended)

Run the main training script:
```bash
python train_biomistral.py
```

This script will:
- Check your system requirements
- Load your full dataset (4,922 samples)
- Set up BioMistral-7B with LoRA fine-tuning
- Train for 2 epochs with memory optimization
- Save the trained model

### Option 2: Advanced Training

For more control over training parameters:
```bash
python biomistral_finetune_optimized.py
```

## Training Details

- **Model**: microsoft/BioMistral-7B
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit quantization for memory efficiency
- **Dataset**: Full DiseaseAndSymptoms_with_Criticality.csv (4,922 samples)
- **Task**: 3-class criticality classification (Mild, Moderate, Critical)
- **Training Time**: ~2-4 hours on a modern GPU

## Using the Trained Model

After training, use the inference script:

```bash
python biomistral_inference.py
```

This will:
- Load your trained BioMistral model
- Process audio input (if available)
- Predict criticality using the fine-tuned model
- Generate audio responses

## Model Outputs

The training will create two directories:
- `./biomistral-medical-criticality/` - Full model and tokenizer
- `./biomistral-medical-criticality-lora/` - LoRA adapters only

## Memory Optimization

The training scripts include several memory optimizations:
- 4-bit quantization
- Gradient checkpointing
- Reduced batch sizes with gradient accumulation
- LoRA fine-tuning (only trains ~1% of parameters)

## Troubleshooting

### Out of Memory Errors
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Reduce `max_length` in tokenization

### Training Too Slow
- Ensure you're using a GPU
- Check CUDA installation
- Consider using a more powerful GPU

### Model Not Loading
- Ensure all dependencies are installed
- Check that the model files were saved correctly
- Verify the model path in inference scripts

## Performance Expectations

BioMistral-7B should provide significantly better performance than the small model:
- Better understanding of medical terminology
- More accurate criticality classification
- Improved handling of complex symptom combinations
- Better generalization to unseen cases

## Next Steps

1. Train the model using `train_biomistral.py`
2. Test with `biomistral_inference.py`
3. Integrate into your existing MediBridge application
4. Monitor performance and retrain if needed

## Support

If you encounter issues:
1. Check the error messages carefully
2. Ensure all requirements are met
3. Verify your dataset format
4. Check GPU memory usage during training
