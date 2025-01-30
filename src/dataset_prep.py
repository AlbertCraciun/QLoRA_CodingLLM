import os
import yaml
from datasets import load_dataset, DatasetDict
from utils import download_github_repos, download_azure_repos

# List of Code File Extensions
FILE_EXTENSIONS = [".py", ".js", ".java", ".cpp", ".cs", ".ts"]

# Load Configurations
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Directories
DATA_DIR = "data/repos"
os.makedirs(DATA_DIR, exist_ok=True)

# Load The Stack dataset
print("Loading 'The Stack' dataset from Hugging Face...")
huggingface_dataset = load_dataset("bigcode/the-stack-smol")

# Download Code Repositories
download_github_repos()
download_azure_repos()

# Process Code Files from Local Repos
def extract_code_files():
    """Extracts .py, .js, .java, .cpp files from local repositories."""
    code_samples = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(tuple(FILE_EXTENSIONS)):
                with open(os.path.join(root, file), "r", errors="ignore") as f:
                    code_samples.append({"text": f.read()})
    return code_samples

local_repo_data = extract_code_files()

# Merge Datasets
final_dataset = DatasetDict({
    "train": huggingface_dataset["train"].select(range(50000)),  # Use 50K samples
    "test": huggingface_dataset["test"].select(range(1000)),     # Use 1K samples
    "custom": local_repo_data  # Add custom repo code
})

# Save Processed Dataset
final_dataset.save_to_disk("processed_dataset")
print("Dataset processing complete!")
