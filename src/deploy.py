from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
pipe = pipeline("text-generation", model="models/qlora_output")

@app.post("/generate")
def generate_code(prompt: str):
    return {"code": pipe(prompt, max_length=150)}
