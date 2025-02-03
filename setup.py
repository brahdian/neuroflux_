from setuptools import setup, find_packages

# Core dependencies required for basic functionality
CORE_DEPS = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "tqdm>=4.65.0",
    "pyyaml>=6.0",
    "jsonschema>=4.17.0",
    "matplotlib>=3.4.0",
    "pandas>=1.3.0",
    "plotly>=5.0.0",
]

# TPU-specific dependencies
TPU_DEPS = [
    "cloud-tpu-client",
    "torch_xla @ git+https://github.com/pytorch/xla",  # Get latest XLA version
]

# Monitoring dependencies
MONITORING_DEPS = [
    "wandb>=0.15.0",
    "psutil>=5.9.0",
    "gputil>=1.4.0",
    "tensorboard>=2.12.0",
]

# Distributed training dependencies
DISTRIBUTED_DEPS = [
    "deepspeed>=0.9.0",
    "torch>=2.0.0",  # Ensure FSDP compatibility
    "accelerate>=0.20.0",
]

setup(
    name="neuroflux",
    version="0.1.0",
    description="Self-Optimizing, Fault-Tolerant Neural Architecture",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="NeuroFlux Team",
    author_email="team@neuroflux.ai",
    url="https://github.com/brahdian/neuroflux_",
    packages=find_packages(),
    python_requires=">=3.8,<3.11",  # TPU support requires Python <3.11
    install_requires=CORE_DEPS,
    extras_require={
        "tpu": TPU_DEPS,
        "monitoring": MONITORING_DEPS,
        "distributed": DISTRIBUTED_DEPS,
        "all": TPU_DEPS + MONITORING_DEPS + DISTRIBUTED_DEPS,
        "dev": [
            "pytest>=7.3.1",
            "black>=23.3.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.3.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
    zip_safe=False,
) 