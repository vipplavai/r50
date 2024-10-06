import subprocess
import sys

def install_packages():
    # List of external packages to be installed
    packages = [
        "paramiko",  # SSH handling
        "psutil",  # System and process utilities
        "flask",  # Web framework for Flask server
        "torch",  # PyTorch
        "transformers",  # Hugging Face Transformers
        "datasets",  # Hugging Face Datasets
        "tokenizers",  # Hugging Face Tokenizers
        "torchvision",  # for some additional utilities, not explicitly imported but often used with torch
        "tqdm",  # Progress bar
        "python-dotenv",  # For .env file handling
        "tensorboard",  # TensorBoard for visualization
        "huggingface_hub",  # Hugging Face Hub API
        "protobuf==3.20.3",  # Specific version of Protobuf required
    ]

    # Iterate through the packages and install each one
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
        else:
            print(f"Successfully installed {package}")

if __name__ == "__main__":
    install_packages()
