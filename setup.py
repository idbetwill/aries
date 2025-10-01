"""
Setup script for Aries Trading Agent

Risk-averse energy trading agent for Colombian market using Reinforcement Learning.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="aries-trading-agent",
    version="1.0.0",
    author="Aries Team",
    author_email="aries@example.com",
    description="Risk-Averse Energy Trading Agent for Colombian Market",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/aries",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aries=aries.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "aries": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "reinforcement-learning",
        "energy-trading",
        "risk-management",
        "colombia",
        "xm",
        "trading-agent",
        "machine-learning",
        "finance"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-repo/aries/issues",
        "Source": "https://github.com/your-repo/aries",
        "Documentation": "https://aries.readthedocs.io/",
    },
)
