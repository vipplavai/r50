from setuptools import setup, find_packages

setup(
	name="ddp_setup",
	version="1.0",
	packages=find_packages(),
	install_requires=[
    	"torch==1.13.0",  # Adjusted to match torchvision requirements
    	"torchvision==0.14.0",
    	"numpy==1.24.2",
    	"tensorboard==2.9.0",
    	"scipy==1.9.3",
    	"scikit-learn==1.0.2",
    	"pandas==1.4.2",
    	"matplotlib==3.6.3",
        "paramiko",
        "datasets",
        "transformers",
        "tqdm",
        "datetime",
        "nltk"
	],
)
