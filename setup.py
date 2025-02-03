from setuptools import setup, find_namespace_packages

packages = [
    'neuroflux',
    'neuroflux.core',
    'neuroflux.system',
    'neuroflux.training',
    'neuroflux.monitoring',
    'neuroflux.utils',
    'neuroflux.evaluation'
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
    packages=packages,
    package_dir={'': '.'},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "jsonschema>=4.17.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "plotly>=5.0.0",
        "wandb>=0.15.0",
        "psutil>=5.9.0",
        "gputil>=1.4.0",
        "tensorboard>=2.12.0",
        "deepspeed>=0.9.0",
        "accelerate>=0.20.0",
        "torch-xla-nightly-cpu; python_version<'3.11'",
        "cloud-tpu-client; python_version<'3.11'"
    ],
    extras_require={
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
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
    zip_safe=False,
) 