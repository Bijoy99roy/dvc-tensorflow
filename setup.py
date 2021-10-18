from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="Bijoy99roy",
    description="A small package for dvc ml pipeline demo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bijoy99roy/dvc-tensorflow",
    author_email="bijoy2014amp@gmail.com",
    packages=["src"],
    python_requires=">=3.6",
    install_requires=[
        'dvc',
        'tensorflow',
        'matplotlib',
        'numpy',
        'pandas',
        'tqdm',
        'pyYAML',
        'boto3'
    ]
)