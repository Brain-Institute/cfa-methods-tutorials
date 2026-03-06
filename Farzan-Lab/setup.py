from setuptools import setup, find_packages

# Version constraints for compatibility:
# - numpy <2.0 for scipy 1.11.4 compatibility
# - scipy <1.16 for statsmodels compatibility
# - scikit-learn >=1.2.2 for existing codebase compatibility

setup(
    name="imbiotype",
    version="0.1.0",
    description="A library for multimodal data integration and biotype analysis",
    author="Raaj Chatterjee",
    author_email="raajc@sfu.ca",
    packages=find_packages(where="lib"),
    package_dir={"": "lib"},
    install_requires=[
        "numpy>=1.26.4,<2.0",
        "pandas>=2.2.3",
        "scikit-learn>=1.2.2",
        "scipy>=1.11.4,<1.16",
        "matplotlib>=3.10.0",
        "cca-zoo>=2.6.0"
    ],
    extras_require={
        "gui": [
            "pyqt5>=5.15.11",
            "mne[full-no-qt]>=1.10.1"
        ]
    },
    python_requires=">=3.11",
) 