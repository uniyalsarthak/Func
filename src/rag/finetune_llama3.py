import json
import os
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer

# --- Device Configuration ---
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")
if DEVICE == "cpu":
    print("WARNING: Fine-tuning on a CPU is extremely slow. It is recommended to use a GPU (CUDA or MPS).")


# --- Configuration ---
# Model and tokenizer
# IMPORTANT: You need a local copy of the model in Hugging Face format.
# The path to this model should be provided here.
# While Ollama is used for serving, the training script needs the raw model files.
# For example, you can download it from Hugging Face Hub: meta-llama/Meta-Llama-3-8B
MODEL_NAME = "llama3:8b"  # Replace with the actual path to your local HF model directory

# Data files
CHAT_LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chat_logs.jsonl')
PROCESSED_LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chat_logs_processed.jsonl')

# Fine-tuning parameters
FINETUNE_THRESHOLD = 10  # Minimum number of new QA pairs to trigger fine-tuning
OUTPUT_DIR = "./llama3-finetuned-lora"


def load_new_qa_pairs():
    """
    Loads new, high-quality QA pairs from the chat log file.
    Filters for entries with "feedback": "up".
    """
    new_pairs = []
    if not os.path.exists(CHAT_LOG_PATH):
        return new_pairs
        
    with open(CHAT_LOG_PATH, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                if log_entry.get("feedback") == "up":
                    new_pairs.append({
                        "instruction": "You are a helpful assistant. Provide a detailed and accurate answer to the user's query.",
                        "input": log_entry["query"],
                        "output": log_entry["answer"]
                    })
            except json.JSONDecodeError:
                print(f"Skipping malformed line: {line.strip()}")
    return new_pairs

def archive_processed_logs():
    """
    Archives the logs that have been used for fine-tuning and clears the original log file.
    """
    original_logs = []
    with open(CHAT_LOG_PATH, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                if log_entry.get("feedback") == "up":
                    original_logs.append(line)
            except json.JSONDecodeError:
                continue

    with open(PROCESSED_LOG_PATH, 'a') as f:
        for line in original_logs:
            f.write(line)

    remaining_logs = []
    with open(CHAT_LOG_PATH, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                if log_entry.get("feedback") != "up":
                    remaining_logs.append(line)
            except json.JSONDecodeError:
                remaining_logs.append(line)

    with open(CHAT_LOG_PATH, 'w') as f:
        for line in remaining_logs:
            f.write(line)

    print(f"Archived {len(original_logs)} processed logs to {PROCESSED_LOG_PATH}")


def format_instruction(sample):
    """
    Formats a sample for the SFTTrainer.
    """
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}
"""

def main():
    """
    Main function to run the fine-tuning pipeline.
    """
    print("Starting fine-tuning process...")

    qa_pairs = load_new_qa_pairs()
    print(f"Found {len(qa_pairs)} new high-quality QA pairs.")

    if len(qa_pairs) < FINETUNE_THRESHOLD:
        print(f"Not enough new QA pairs to start fine-tuning. Need at least {FINETUNE_THRESHOLD}.")
        return

    dataset = Dataset.from_list(qa_pairs)
    print("Sufficient data found. Starting model fine-tuning...")

    # --- QLoRA Configuration ---
    # Using QLoRA for maximum efficiency on CUDA, and LoRA on MPS/CPU
    
    # 1. LoRA Config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 2. Quantization Config (for CUDA only)
    bnb_config = None
    if DEVICE == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto" # Automatically places layers on available devices
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2, # Reduced for better compatibility
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=10,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        report_to="none",
        fp16= (DEVICE == "cuda"), # fp16 is for CUDA
        # bf16=False, # Set to True if your hardware supports it
    )

    # Trainer configuration
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config, # Pass the LoRA config here
        formatting_func=format_instruction,
        max_seq_length=512,
        packing=True, # Pack sequences for higher efficiency
    )

    # Start training
    try:
        print("Training...")
        trainer.train()
        print("Training finished.")

        print(f"Saving model to {OUTPUT_DIR}...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Model and tokenizer saved.")

        # 5. Archive the processed logs only on successful training
        archive_processed_logs()
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Logs will not be archived. The script will attempt to re-train on the next run.")

if __name__ == "__main__":
    main()