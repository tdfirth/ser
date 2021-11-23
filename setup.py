from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.readlines()


setup(
    name="ser",
    version="0.0.1",
    author="Thomas Firth & Claudia Vanea",
    description="MNIST for SER",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tdfirth/ser",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires=">=3.9",
    entry_points={"console_scripts": ["ser = bin.cli:main"]},
)
