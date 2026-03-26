from setuptools import setup, find_packages

setup(
    name="Metacognitive-Medical-Digital-Twins",
    version="1.0.0",
    author="Ahmed Soliman",
    author_email="ahmed.soliman@ufl.edu",
    description="Metacognitive Medical Digital Twin with Triple-Stream Reasoning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AhmedSSoliman/Metacognitive-Medical-Digital-Twins-MDT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    python_requires=">=3.14",
    install_requires=open("requirements.txt").read().splitlines(),
)