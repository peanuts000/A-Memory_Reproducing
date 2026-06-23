from setuptools import setup, find_packages

setup(
    name="amem",
    version="1.0.0",
    description="A-Mem: 面向 LLM Agent 的智能记忆系统 - 论文忠实复现",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="A-Memory Reproducing",
    url="https://github.com/peanuts000/A-Memory_Reproducing",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "sentence-transformers>=2.2.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "openai>=1.0.0",
        "litellm>=1.0.0",
        "rouge-score>=0.1.2",
        "nltk>=3.8.0",
        "python-dotenv>=1.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
