import os
from setuptools import setup, find_packages

# Import version information
about = {}
with open(os.path.join(os.path.dirname(__file__), "version.py"), encoding="utf-8") as f:
    exec(f.read(), about)

# Function to safely read README.md
def read_readme():
    try:
        if os.path.exists("README.md"):
            with open("README.md", encoding="utf-8") as f:
                return f.read()
        return ""
    except:
        return about["__description__"]

setup(
    name="Arthashila",
    version=about["__version__"],
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.24.0",
        "streamlit-option-menu>=0.3.2",
        "psutil>=5.9.0",
        "plotly>=5.13.0",
        "pandas>=1.3.0",
        "tabulate>=0.9.0",
        "numpy>=1.22.0",
        "scikit-learn>=1.0.2",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "joblib>=1.1.0",
        "threadpoolctl>=3.0.0"
    ],
    author=about["__author__"],
    author_email=about["__email__"],
    description=about["__description__"],
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/arthashila/arthashila",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 