import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

# Load Configurations
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load Model & Tokenizer
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    torch_dtype=torch.float16,
    load_in_4bit=True,  # Enable QLoRA
    trust_remote_code=True  # Required for some HF models
)
tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)

# Load Processed Dataset
dataset = load_from_disk("processed_dataset")

# Apply QLoRA
lora_config = LoraConfig(
    r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
    target_modules=config["lora_target_modules"], bias=config["lora_bias"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Training Arguments
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_steps=10,
    save_steps=500,
    per_device_train_batch_size=config["batch_size"],
    num_train_epochs=config["epochs"],
    learning_rate=config["learning_rate"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    fp16=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# Train Model
print("Training started...")
trainer.train()
print("✅ Training complete! Model saved in 'models/qlora_output'")

# Save Tokenizer
tokenizer.save_pretrained(config["output_dir"])
