from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer
import torch
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.training_args import TrainingArguments

# Configuration
MODEL_NAME = "llama3.2:3b"  # Use 8B for local fine-tuning
DATASET_PATH = "qa-data.json"
OUTPUT_DIR = "./llama3-finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load and preprocess dataset
dataset = load_dataset("json", data_files=DATASET_PATH)
if isinstance(dataset, dict) and "train" in dataset:
    dataset = dataset["train"]

def format_instruction(sample):
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}
"""

# Quantization configuration (4-bit for GPU efficiency)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config if DEVICE == "cuda" else None,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=DEVICE == "cuda",
    logging_steps=10,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    report_to="none"
)

# Trainer configuration
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=format_instruction,
)
# Start training
trainer.train()

# Save model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)