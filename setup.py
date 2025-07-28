"""
Setup script for TUMTraffiX-qwen train and evaluation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tumtraffic-qa",
    version="1.0.0",
    author="TUMTraffic Team",
    author_email="contact@tumtraffic.com",
    description="Traffic Video Question Answering Dataset and Evaluation Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LLaVA-VL/LLaVA-NeXT",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "tumtraffic-eval=eval_tumtraf:main",
        ],
    },
) 