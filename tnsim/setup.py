"""Setup script for TNSIM (Theory of Zero Sum of Infinite Sets)."""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Package version
__version__ = "1.0.0"

setup(
    name="tnsim",
    version=__version__,
    author="TNSIM Team",
    author_email="tnsim@example.com",
    description="Theory of Zero Sum of Infinite Sets - library for working with compensated infinite sets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tnsim/tnsim",
    project_urls={
        "Bug Tracker": "https://github.com/tnsim/tnsim/issues",
        "Documentation": "https://tnsim.readthedocs.io/",
        "Source Code": "https://github.com/tnsim/tnsim",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
            "pre-commit>=3.6.0",
        ],
        "docs": [
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=1.3.0",
            "nbsphinx>=0.9.3",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.8",
            "matplotlib>=3.8.2",
            "seaborn>=0.13.0",
            "plotly>=5.17.0",
            "ipywidgets>=8.1.1",
        ],
        "balansis": [
            # "balansis>=1.0.0",  # Uncomment when Balansis becomes available
        ],
        "performance": [
            "dask[complete]>=2023.11.0",
            "ray[default]>=2.8.0",
            "numba>=0.58.1",
        ],
        "monitoring": [
            "prometheus-client>=0.19.0",
            "grafana-api>=1.0.3",
            "influxdb-client>=1.38.0",
        ],
        "all": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
            "pre-commit>=3.6.0",
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=1.3.0",
            "nbsphinx>=0.9.3",
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.8",
            "matplotlib>=3.8.2",
            "seaborn>=0.13.0",
            "plotly>=5.17.0",
            "ipywidgets>=8.1.1",
            "dask[complete]>=2023.11.0",
            "ray[default]>=2.8.0",
            "numba>=0.58.1",
            "prometheus-client>=0.19.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tnsim-server=tnsim.api.main:main",
            "tnsim-cli=tnsim.cli.main:main",
            "tnsim-migrate=tnsim.database.migrate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "tnsim": [
            "migrations/*.sql",
            "examples/*.py",
            "examples/*.ipynb",
            "docs/*.md",
        ],
    },
    zip_safe=False,
    keywords=[
        "mathematics",
        "infinite-sets",
        "zero-sum",
        "compensation",
        "numerical-stability",
        "machine-learning",
        "attention-mechanism",
        "balansis",
        "fastapi",
        "pytorch",
    ],
)

# Additional package information
if __name__ == "__main__":
    print(f"TNSIM (Theory of Zero Sum of Infinite Sets) version {__version__}")
    print("Theory of Zero Sum of Infinite Sets")
    print("To install: pip install .")
    print("For development: pip install -e .[dev]")
    print("For all dependencies: pip install -e .[all]")