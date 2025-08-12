- implemented async rag base based pipeline
- in memory cache support
- Fine tuning the LLM after certain number of repsonses
- using vector db && ollama locall llm
- chunking and embeddings done again make it more effiecint , batch size - 5000 , cleaned noise and irrelvant stuff
- along with recursive search
- Feedback taken and stored , along with that every qa-repsonse is stored for fine tune

- 1.  Parameter-Efficient Fine-Tuning (PEFT): I will switch from full fine-tuning to QLoRA (Quantized Low-Rank
      Adaptation). Instead of retraining all 8 billion parameters, this method freezes the main model and only
      trains a very small number of "adapter" layers. This is the industry-standard best practice for drastically
      reducing memory usage (by up to 70-80%) and increasing training speed.

- lora adapters

2. How to Use the Fine-Tuned Model with Ollama

This is the most important part of the process. The fine-tuning script does not output a full, ready-to-use
model. For efficiency, it only outputs the "changes" to the model, which are called LoRA adapters.

Where is the model stored?
The script saves these LoRA adapters into the directory specified by OUTPUT_DIR, which is
./llama3-finetuned-lora.

The Workflow: Merge, then Import
To use this with Ollama, you must first merge these small adapter files back into the original base model.
This creates a new, standalone, fine-tuned model.

I will provide a new script to handle this merging process.

Step 1: Create the Merge Script
I will create a new file, src/rag/merge_lora_weights.py, that will combine the base model and your LoRA
adapter.

✦ Step 2: Your Full Workflow
Here is your complete, end-to-end process:

1.  Fine-Tune: Run the training script as before.
    1 python src/rag/finetune_llama3.py
2.  Merge: After training is complete, run the new merge script.
    1 python src/rag/merge_lora_weights.py
3.  Import to Ollama: Create a file named Modelfile in your project root with this line:
    1 FROM ./llama3-finetuned-merged
    Then, run the Ollama create command:
    1 ollama create my-custom-llama3 -f Modelfile
4.  Run: You can now run your personalized model in Ollama!
