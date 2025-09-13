"""
Setup script for OmicsFormer package.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="omicsformer",
    version="0.1.0",
    author="Shiva Prasad",
    author_email="your.email@domain.com",
    description="Advanced Multi-Omics Integration with Transformers",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/omicsformer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "umap-learn>=0.5.0",
        "tqdm>=4.60.0",
        "scipy>=1.7.0",
        "networkx>=2.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "pre-commit>=2.15.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "wandb": [
            "wandb>=0.12.0",
        ],
        "full": [
            "wandb>=0.12.0",
            "plotly>=5.0.0",
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "omicsformer-example=examples.complete_analysis_example:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="multi-omics, transformers, bioinformatics, machine learning, pytorch, attention, genomics, proteomics, metabolomics",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/omicsformer/issues",
        "Source": "https://github.com/yourusername/omicsformer",
        "Documentation": "https://omicsformer.readthedocs.io/",
    },
)