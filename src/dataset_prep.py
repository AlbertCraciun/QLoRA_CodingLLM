import os
import yaml
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from utils import download_github_repos, download_azure_repos

# List of Code File Extensions
FILE_EXTENSIONS = [".py", ".js", ".java", ".cpp", ".cs", ".ts"]

# Load Configurations
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Directories
DATA_DIR = "data/repos"
os.makedirs(DATA_DIR, exist_ok=True)

# Load The Stack (sau alt dataset definit în `config.yaml`)
print(f"Loading dataset '{config['dataset_path']}' from Hugging Face...")
huggingface_dataset = load_dataset(config["dataset_path"])

# Download Code Repositories
download_github_repos()
download_azure_repos()

# Process Code Files from Local Repos
def extract_code_files():
    """Extracts .py, .js, .java, .cpp, .cs, .ts files from local repositories."""
    code_samples = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(tuple(FILE_EXTENSIONS)):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, "r", errors="ignore") as f:
                        code = f.read()
                        code_samples.append({"text": code})
                except Exception as e:
                    print(f"Error reading file {full_path}: {e}")
    return code_samples

local_repo_data = extract_code_files()

# Convert local_repo_data to a Hugging Face Dataset
custom_dataset = None
if len(local_repo_data) > 0:
    custom_dataset = Dataset.from_list(local_repo_data)
else:
    print("No local code files found. Skipping custom dataset creation.")

# Select a smaller subset for train/test to avoid huge memory usage
train_dataset = huggingface_dataset["train"].select(range(min(50000, len(huggingface_dataset["train"]))))
test_dataset  = huggingface_dataset["test"].select(range(min(1000, len(huggingface_dataset["test"]))))

# If we have custom data, we'll combine it with the training set
if custom_dataset is not None and len(custom_dataset) > 0:
    train_dataset = concatenate_datasets([train_dataset, custom_dataset])
    print(f"Added {len(custom_dataset)} samples from local repos to train set.")

final_dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# Save Processed Dataset
final_dataset.save_to_disk("processed_dataset")
print("Dataset processing complete! Final dataset:")
print(final_dataset)
