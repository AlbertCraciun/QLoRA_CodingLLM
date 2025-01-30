import os
import git
import requests
import yaml

# Load Configurations
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Define Directories
DATA_DIR = "data/repos"
os.makedirs(DATA_DIR, exist_ok=True)

def clone_repo(url, local_dir):
    """ Clone a Git repository. """
    try:
        git.Repo.clone_from(url, local_dir)
        print(f"✅ Cloned: {url}")
    except Exception as e:
        print(f"❌ Failed to clone {url}: {e}")

def download_github_repos():
    """ Download GitHub repositories from BigCode. """
    print("📥 Downloading GitHub repositories from BigCode...")
    response = requests.get(config["github_api_url"])

    if response.status_code == 200:
        repos = response.json()
        for repo in repos:
            repo_url = repo["clone_url"]
            repo_name = repo["name"]
            clone_repo(repo_url, os.path.join(DATA_DIR, repo_name))
    else:
        print(f"❌ Failed to fetch GitHub repos: {response.text}")

def download_azure_repos():
    """ Download private Azure repositories. """
    if not config["azure_repos"]:
        print("⚠️ Azure repo download is disabled in config.yaml")
        return

    print("📥 Downloading Azure DevOps repositories...")
    AZURE_ORG = config["azure_organization"]
    AZURE_PROJECT = config["azure_project"]
    AZURE_TOKEN = os.getenv("AZURE_PAT")  # Personal Access Token from .env

    if not AZURE_TOKEN:
        print("❌ Missing Azure Personal Access Token (AZURE_PAT) in .env file")
        return

    headers = {"Authorization": f"Basic {AZURE_TOKEN}"}
    url = f"https://dev.azure.com/{AZURE_ORG}/{AZURE_PROJECT}/_apis/git/repositories?api-version=6.0"

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        repos = response.json()["value"]
        for repo in repos:
            repo_url = repo["remoteUrl"]
            repo_name = repo["name"]
            clone_repo(repo_url, os.path.join(DATA_DIR, repo_name))
    else:
        print(f"❌ Failed to fetch Azure repos: {response.text}")
