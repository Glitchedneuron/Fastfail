from setuptools import setup, find_packages

setup(
    name="fastfail-llm",
    version="1.0.0",
    description="Domain-specific LLM/SLM training framework",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "tokenizers>=0.19.0",
        "accelerate>=0.28.0",
        "peft>=0.10.0",
        "trl>=0.8.0",
        "bitsandbytes>=0.43.0",
        "optuna>=3.6.0",
        "safetensors>=0.4.3",
        "rich>=13.7.0",
        "click>=8.1.7",
        "questionary>=2.0.1",
        "psutil>=5.9.8",
        "loguru>=0.7.2",
        "requests>=2.31.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "tqdm>=4.66.0",
        "PyYAML>=6.0.1",
    ],
    entry_points={
        "console_scripts": [
            "fastfail=main:cli",
        ],
    },
)
