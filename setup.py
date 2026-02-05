from setuptools import find_packages, setup

setup(
    name="rlpi",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # NOTE: torch is NOT listed here - it must be pre-installed with CUDA support
        # Install via: pip install torch --index-url https://download.pytorch.org/whl/cu128
        # Core dependencies
        "transformers==4.57.0",
        # External dependency: AgentDojo (task suites + pipeline abstractions)
        "agentdojo",
        "numpy>=1.19.0",
        "matplotlib==3.10.3",
        "tqdm==4.67.1",
        "hydra-core>=1.3.0",
        "hydra-submitit-launcher>=1.2.0",
        "typing-extensions>=4.0.0",
        # TRL ecosystem
        "trl==0.24.0",
        "peft>=0.6.0",
        "accelerate>=0.24.0",
        "datasets>=2.14.0",
        "bitsandbytes>=0.41.0",
        # Tokenizers and formats
        "sentencepiece",
        "protobuf",
        # OpenAI API
        "openai>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.900",
            "pre-commit>=2.15",
            "types-PyYAML>=6.0",
        ]
    },
    python_requires=">=3.10",
)
