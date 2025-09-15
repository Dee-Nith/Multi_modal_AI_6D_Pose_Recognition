#!/usr/bin/env python3
"""
Setup script for AI 6D Pose Recognition System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-6d-pose-recognition",
    version="1.0.0",
    author="[Your Name]",
    author_email="[your.email@example.com]",
    description="A comprehensive computer vision system for 6D pose estimation using YOLOv8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AI_6D_Pose_recognition",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-pose-recognition=main:main",
        ],
    },
)
