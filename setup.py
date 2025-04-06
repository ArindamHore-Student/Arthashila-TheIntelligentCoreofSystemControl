import os
from setuptools import setup, find_packages

setup(
    name="SimpleRTOS",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.24.0",
        "streamlit-option-menu>=0.3.2",
        "psutil>=5.9.0",
        "plotly>=5.13.0",
        "tabulate>=0.9.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive system monitoring and task management application",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SimpleRTOS",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 