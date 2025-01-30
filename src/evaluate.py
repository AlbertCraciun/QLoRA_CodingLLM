import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Fine-Tuned Model
model_path = "models/qlora_output"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Test Prompt
prompt = "Write a Python function to merge two sorted lists."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate Code
output = model.generate(**inputs, max_length=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
