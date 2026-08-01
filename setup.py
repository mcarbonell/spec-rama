from setuptools import setup, find_packages

setup(
    name="spec-rama",
    version="0.1.0",
    description="Permutated Spectral PEFT Framework: RAMA modulation with DCT, FWHT, and Wavelet Core adaptation",
    author="Antigravity Research",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
