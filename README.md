## QLoRA_CodingLLM

This project demonstrates how to fine-tune the **DeepSeek-R1** model using **QLoRA** for code generation.
It supports datasets from BigCode (GitHub), Azure DevOps, and The Stack (Hugging Face), along with a complete flow for training, evaluation, and deployment with FastAPI.

---

### **Key Features**
- **Fine-tune DeepSeek-R1** on a custom dataset (GitHub, Azure, The Stack)
- **Uses QLoRA (4-bit quantization)** to reduce memory requirements
- **Extract and process code** from GitHub, Azure DevOps, and Hugging Face
- **Full deployment** with FastAPI (containerized)
- **Automation** with a single `docker-compose up --build`

---

### **Project Structure**
```
.
│── docker-compose.yml       # Config for running in Docker containers
│── setup.sh                 # Dependency installation script (for local run)
│── requirements.txt         # Python libraries list
│── config.yaml              # Configuration file (model, dataset, parameters)
│── .env                     # Environment variables (Azure, tokens); ignored by commit
│── data/                    # Folder for code downloaded from GitHub/Azure
│── models/                  # Folder for fine-tuned models
│── src/
│   │── dataset_prep.py      # Script for dataset preparation (GitHub + The Stack)
│   │── train.py             # QLoRA fine-tuning for DeepSeek-R1
│   │── evaluate.py          # Script for evaluating the trained model
│   │── deploy.py            # Deployment with FastAPI
│   │── utils.py             # Utility functions (GitHub, Azure clone)
│── README.md                # Project documentation
```

---

### **Installation & Setup**

#### **1. Clone this repository**
```bash
git clone https://github.com/your-repo/QLoRA_CodingLLM.git
cd QLoRA_CodingLLM
```

#### **2. (Optional) Local Run (Without Docker)**
If you prefer to run locally, **without containers**, you can use the `setup.sh` script which:
- Installs Python dependencies
- Creates and activates a `venv` virtual environment

```bash
chmod +x setup.sh
./setup.sh

# Activate the virtual environment
source venv/bin/activate
```
> **Note**: In this mode, you will run scripts with `python src/dataset_prep.py`, `python src/train.py`, etc.

#### **3. Configure `.env` (For Azure)**
If you want to download code from Azure DevOps, create an **`.env`** file with your access data:
```ini
AZURE_PAT=your_personal_access_token
```
In `config.yaml` you can enable/disable downloading from Azure via `azure_repos: true/false`.

---

### **One-Click Execution (Docker)**
To run the entire workflow (dataset preparation, training, evaluation, FastAPI deployment) with a single command:

```bash
docker-compose up --build
```

- **`qlora-training`** will prepare the data (if not already prepared), then train the model with QLoRA.
- **`qlora-api`** will start after the model is trained, launching a FastAPI server at `http://localhost:8000`.

If you only want to run the training:
```bash
docker-compose up --build qlora-training
```

If you only want the API (assuming the model is already trained and saved in `models/qlora_output`):
```bash
docker-compose up --build qlora-api
```

---

### **Optional Run: GPU Check**
To check if the GPU is available in the container, you can use:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-devel-ubuntu22.04 nvidia-smi
```
If you have the drivers and `nvidia-docker2` installed, you’ll see GPU details.

---

### **Manual Local Run**
If you do not want to use containers:
1. Install dependencies with `setup.sh`.
2. Run them in sequence:
   ```bash
   # 1) Prepare the dataset
   python src/dataset_prep.py

   # 2) Train the model with QLoRA
   python src/train.py

   # 3) Evaluate the model
   python src/evaluate.py

   # 4) Deploy with FastAPI on port 8000
   uvicorn src.deploy:app --reload --host 0.0.0.0 --port 8000
   ```
---

### **Customizing for Other Models**
This project can be adapted to any LLM instead of **DeepSeek-R1**.

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

#### **Compatible Models**
- **Meta LLaMA 2**
- **Code LLaMA**
- **Mistral-7B**
- **StarCoder**
- **GPT-J / GPT-NeoX**
etc.

### **FAQ**
1. **Can I run it on a local RTX 4090?**
   Yes, QLoRA is perfect for 24GB VRAM. Performance is higher on A100/H100, but 4090 is sufficient.

2. **What datasets can I use?**
   - The Stack (Hugging Face)
   - GitHub Repos (BigCode)
   - Private Azure DevOps Repos

3. **How do I deploy to production?**
   - Use Docker and FastAPI
   - You can move the Docker image to AWS, GCP, RunPod, etc.
   - You can upload the fine-tuned model to Hugging Face with:
     ```python
     model.push_to_hub("my-fine-tuned-deepseek")
     ```

4. **Why can’t I see local code in the dataset?**
   In `src/dataset_prep.py`, local files are transformed into a `Dataset` and concatenated with `train`. Check the logs to ensure that files are being read correctly.

---

### **Useful Resources**
- [Hugging Face: The Stack](https://huggingface.co/datasets/bigcode/the-stack)
- [DeepSeek AI](https://huggingface.co/deepseek-ai)
- [Hugging Face: Fine-Tuning Guide](https://huggingface.co/docs/transformers/training)
- [QLoRA Paper (arXiv)](https://arxiv.org/abs/2305.14314)
- [Finetuning Code LLM (Hugging Face Cookbook)](https://huggingface.co/learn/cookbook/fine_tuning_code_llm_on_single_gpu)
- [Code LLaMA](https://huggingface.co/codellama)

---

#### **Kaggle**
If you want to fine-tune a model using **Kaggle** adn HuggingFace, you can acces [this notebook](https://www.kaggle.com/code/albertcrciun/unsloth-llm-fine-tunning-untested)

---

## **Contributors**
- [Albert Cristian Crăciun](https://www.linkedin.com/in/albertc1078/) - Software Engineer
