## QLoRA_CodingLLM

Acest proiect demonstrează cum să fine-tune-uiesti modelul **DeepSeek-R1** folosind **QLoRA** pentru generarea de cod.
Suportă dataset-uri din BigCode (GitHub), Azure DevOps și The Stack (Hugging Face), precum și un flux complet de pregătire, evaluare și deploy cu FastAPI.

---

### **Caracteristici principale**
- **Fine-tune DeepSeek-R1** pe un dataset personalizat (GitHub, Azure, The Stack)
- **Folosește QLoRA (4-bit quantization)** pentru a reduce cerințele de memorie
- **Extrage și prelucrează cod** din GitHub, Azure DevOps și Hugging Face
- **Deploy complet** cu FastAPI (containerizat)
- **Automatizare** cu un singur `docker-compose up --build`

---

### **Structura proiectului**
```
.
│── docker-compose.yml       # Config pentru rularea în containere Docker
│── setup.sh                 # Script de instalare dependențe (pentru rulare locală)
│── requirements.txt         # Lista de librării Python
│── config.yaml              # Fisier de configurare (model, dataset, parametri)
│── .env                     # Variabile de mediu (Azure, token-uri); ignorat la commit
│── data/                    # Folder pentru cod descărcat din GitHub/Azure
│── models/                  # Folder pentru modelele fine-tunate
│── src/
│   │── dataset_prep.py      # Script de pregătire a dataset-ului (GitHub + The Stack)
│   │── train.py             # Fine-tuning QLoRA pentru DeepSeek-R1
│   │── evaluate.py          # Script de evaluare a modelului antrenat
│   │── deploy.py            # Deploy cu FastAPI
│   │── utils.py             # Funcții utilitare (clone GitHub, Azure)
│── README.md                # Documentația proiectului
```

---

### **Instalare & Setup**

#### **1. Clonează acest repository**
```bash
git clone https://github.com/your-repo/QLoRA_CodingLLM.git
cd QLoRA_CodingLLM
```

#### **2. (Opțional) Rulare locală (Fără Docker)**
Dacă preferi să rulezi local, **fără containere**, poți utiliza scriptul `setup.sh` care:
- Instalează dependențe Python
- Creează și activează un mediu virtual `venv`

```bash
chmod +x setup.sh
./setup.sh

# Activează mediul virtual
source venv/bin/activate
```
> **Notă**: În acest mod, vei rula scripturile cu `python src/dataset_prep.py`, `python src/train.py`, etc.

#### **3. Configurare `.env` (Pentru Azure)**
Dacă vrei să descarci cod din Azure DevOps, creează un fișier **`.env`** cu datele de acces:
```ini
AZURE_PAT=your_personal_access_token
```
În `config.yaml` poți activa/dezactiva descărcarea din Azure prin `azure_repos: true/false`.

---

### **One-Click Execution (Docker)**
Pentru a rula întregul flux (pregătire dataset, antrenare, evaluare, deploy FastAPI) dintr-o singură comandă:

```bash
docker-compose up --build
```

- **`qlora-training`** va pregăti datele (dacă nu sunt deja pregătite), apoi va antrena modelul cu QLoRA.
- **`qlora-api`** va porni după ce modelul e antrenat, lansând un server FastAPI la `http://localhost:8000`.

Dacă vrei să rulezi doar antrenarea:
```bash
docker-compose up --build qlora-training
```

Dacă vrei doar API-ul (presupunând că modelul e deja antrenat și salvat în `models/qlora_output`):
```bash
docker-compose up --build qlora-api
```

---

### **Rulare opțională: GPU verificare**
Pentru a verifica dacă GPU-ul este disponibil în container, poți folosi:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-devel-ubuntu22.04 nvidia-smi
```
Dacă ai instalat driverele și `nvidia-docker2`, vei vedea detalii despre GPU.

---

### **Rulare locală (Manual)**
Dacă nu dorești să folosești containere:
1. Instalează dependențele cu `setup.sh`.
2. Rulează pe rând:
   ```bash
   # 1) Pregătește datasetul
   python src/dataset_prep.py

   # 2) Antrenează modelul cu QLoRA
   python src/train.py

   # 3) Evaluează modelul
   python src/evaluate.py

   # 4) Deploy cu FastAPI pe portul 8000
   uvicorn src.deploy:app --reload --host 0.0.0.0 --port 8000
   ```
---

### **Personalizare pentru alte modele**
Acest proiect poate fi adaptat pentru orice LLM, în loc de **DeepSeek-R1**.

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

#### **Modele compatibile**
- **Meta LLaMA 2**
- **Code LLaMA**
- **Mistral-7B**
- **StarCoder**
- **GPT-J / GPT-NeoX**
etc.

### **FAQ**
1. **Pot rula pe un RTX 4090 local?**
   Da, QLoRA e perfect pentru 24GB VRAM. Performanța e mai mare pe A100/H100, dar 4090 e suficient.

2. **Ce seturi de date pot folosi?**
   - The Stack (Hugging Face)
   - Repos GitHub (BigCode)
   - Repos private Azure DevOps

3. **Cum fac deploy în producție?**
   - Folosește Docker și FastAPI
   - Poți muta imaginea Docker pe AWS, GCP, RunPod etc.
   - Poți urca modelul fine-tunat pe Hugging Face cu:
     ```python
     model.push_to_hub("my-fine-tuned-deepseek")
     ```

4. **De ce nu se vede codul local în dataset?**
   În `src/dataset_prep.py`, fișierele locale sunt transformate în `Dataset` și concatenate cu `train`. Verifică log-urile să te asiguri că fișierele sunt citite corect.

---

### **Resurse utile**
- [Hugging Face: The Stack](https://huggingface.co/datasets/bigcode/the-stack)
- [DeepSeek AI](https://huggingface.co/deepseek-ai)
- [Hugging Face: Fine-Tuning Guide](https://huggingface.co/docs/transformers/training)
- [QLoRA Paper (arXiv)](https://arxiv.org/abs/2305.14314)
- [Finetuning Code LLM (Hugging Face Cookbook)](https://huggingface.co/learn/cookbook/fine_tuning_code_llm_on_single_gpu)
- [Code LLaMA](https://huggingface.co/codellama)

---

### **Contribuitori**
- [Albert Cristian Crăciun](https://www.linkedin.com/in/albertc1078/) - Developer