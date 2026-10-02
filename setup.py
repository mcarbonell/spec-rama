from setuptools import find_packages, setup

setup(
    name="spec-rama",
    version="0.2.0",
    description="Permuted Spectral PEFT Framework: RAMA Modulation with 2D Wavelet, DCT, and Walsh-Hadamard Core Adaptation",
    author="Mario Raúl Carbonell Martínez",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "benchmarks": [
            "transformers>=4.38.0",
            "datasets>=2.16.0",
            "accelerate>=0.27.0",
            "scipy>=1.10.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "ruff>=0.2.0",
            "mypy>=1.8.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
