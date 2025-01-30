## Fine-Tuning DeepSeek-R1 with QLoRA

This project enables fine-tuning the **DeepSeek-R1** model using **QLoRA**
to optimize code generation performance.
It supports datasets from BigCode (GitHub repos), Azure private repositories,
and The Stack (Hugging Face).

## **Project Structure**
```
.
│── docker-compose.yml       # Docker setup for cloud/local deployment
│── setup.sh                 # Automated setup script for dependencies
│── requirements.txt         # List of Python dependencies
│── config.yaml              # Config file for easy customization
│── .env                     # Environment variables for Azure & secrets (ignored)
│── data/                    # Folder to store downloaded code repos
│── models/                  # Folder to save trained models
│── src/
│   │── train.py             # Fine-tuning script for DeepSeek-R1
│   │── dataset_prep.py      # Script to prepare and clean dataset
│   │── evaluate.py          # Script to test trained model
│   │── deploy.py            # FastAPI deployment script
│   │── utils.py             # Helper functions (GitHub/Azure repo extraction)
│── README.md                # Documentation on how to run everything
```

## **Features**
- **Fine-tune DeepSeek-R1 on your dataset**
- **Uses QLoRA (4-bit quantization) for efficient training**
- **Supports extracting code from GitHub, Azure DevOps, and Hugging Face**
- **Deploys the trained model using FastAPI**
- **Fully containerized (Docker + GPU acceleration)**
- **Supports one-click execution for full automation**

## **Installation & Setup**
### **1. Clone the Repository**
```bash
git clone https://github.com/your-repo/QLoRA_CodingLLM.git
cd QLoRA_CodingLLM
```

### **2. Install Dependencies (Local)**
```bash
chmod +x setup.sh
./setup.sh
```

### **3. Configure `.env` File (For Azure)**
Create a **`.env`** file to store **Azure credentials**.
```ini
AZURE_PAT=your_personal_access_token
```

## **One-Click Execution**
To run everything (dataset preparation, training, evaluation, and deployment)
**with a single command**:
```bash
docker-compose up --build
```
This command will:
1. **Prepare the dataset** by downloading and formatting repositories.
2. **Train DeepSeek-R1** using QLoRA.
3. **Evaluate the trained model**.
4. **Deploy the model using FastAPI**.

If no errors occur, the model will be **fully fine-tuned and deployed** when you return.

## **Running Options: Local vs Cloud**
### **Run Locally (Manual Execution)**
```bash
chmod +x setup.sh
./setup.sh
python src/dataset_prep.py
python src/train.py
python src/evaluate.py
uvicorn src.deploy:app --reload --host 0.0.0.0 --port 8000
```

### **Run in a Cloud Environment (Docker)**
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
docker-compose up --build
docker-compose up qlora-training
docker-compose up qlora-api
```

## **Adapting to Other Models**
This project can be modified to fine-tune **any LLM** instead of **DeepSeek-R1**.

### **Where to Change the Model?**
1. **`config.yaml`**
```yaml
model_name: "huggingface-model-name-here"
```
2. **`src/train.py`**
```python
model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    torch_dtype=torch.float16,
    load_in_4bit=True,
    trust_remote_code=True
)
```
3. **`src/evaluate.py`**
```python
model_path = "models/your-model-name"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
```
4. **`src/deploy.py`**
```python
pipe = pipeline("text-generation", model="models/your-model-name")
```
### **Compatible Models**
- **Meta LLaMA 2**
- **Code LLaMA**
- **Mistral-7B**
- **StarCoder**
- **GPT-J/GPT-NeoX**

## **Configuration File (`config.yaml`)**
Modify **`config.yaml`** to customize model training:
```yaml
model_name: "deepseek-ai/DeepSeek-R1"
dataset_path: "bigcode/the-stack-smol"
epochs: 3
batch_size: 2
learning_rate: 0.0002
gradient_accumulation_steps: 4
azure_organization: "your-azure-org"
azure_project: "your-azure-project"
```

## **FAQ**
### **Can I run this on a local GPU (RTX 4090)?**
Yes! **QLoRA allows fine-tuning on 24GB VRAM GPUs.**
For best performance, use **A100 / H100 / RTX 4090**.

### **What datasets does this support?**
- **The Stack** (Hugging Face)
- **GitHub repos (BigCode)**
- **Azure DevOps private repos**

### **How do I deploy the model in production?**
- Use **Docker** and host via **FastAPI**
- Deploy on **AWS, GCP, or RunPod**
- Convert model to **HF Hub** using:
```python
model.push_to_hub("my-fine-tuned-deepseek")
```

## **Resources**
- [Hugging Face: The Stack](https://huggingface.co/datasets/bigcode/the-stack)
- [DeepSeek AI](https://huggingface.co/deepseek-ai)
- [Hugging Face: Fine-Tuning Guide](https://huggingface.co/docs/transformers/training)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Fine-Tuning Large LLMs](https://huggingface.co/learn/cookbook/fine_tuning_code_llm_on_single_gpu)
- [Code LLaMA](https://huggingface.co/codellama)

## **Contributors**
- [Albert Cristian Crăciun](https://www.linkedin.com/in/albertc1078/) - Developer

## **License**
This project is **open-source (MIT License).**