import torch
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
# IMPORTANT: You must provide the path to your local, original base model.
# This should be the same model you used for fine-tuning.
BASE_MODEL_PATH = "llama3:8b"  # Replace with the actual path to your local HF model directory

# Path to the LoRA adapter from the fine-tuning script
LORA_ADAPTER_PATH = "./llama3-finetuned-lora"

# Path to save the final, merged model
MERGED_MODEL_OUTPUT_PATH = "./llama3-finetuned-merged"

# --- Merge Process ---
print(f"Loading base model from: {BASE_MODEL_PATH}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)

print(f"Loading LoRA adapter from: {LORA_ADAPTER_PATH}")
# Load the PEFT model with the LoRA adapter
model_with_adapter = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER_PATH,
)

print("Merging adapter weights with the base model...")
merged_model = model_with_adapter.merge_and_unload()
print("Merge complete.")

print(f"Saving merged model to {MERGED_MODEL_OUTPUT_PATH}...")
os.makedirs(MERGED_MODEL_OUTPUT_PATH, exist_ok=True)
merged_model.save_pretrained(MERGED_MODEL_OUTPUT_PATH)

# Also save the tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_OUTPUT_PATH)

print("\nMerged model and tokenizer saved successfully.")
print(f"You can now import this model into Ollama using the directory: {MERGED_MODEL_OUTPUT_PATH}")
