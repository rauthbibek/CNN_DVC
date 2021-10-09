from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="rauthbibek",
    description="A small package for dvc cnn pipeline demo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rauthbibek/CNN_DVC.git",
    author_email="rauth.bibek66@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        'dvc',
        'pandas',
        'numpy'
        'tensorflow',
        'boto3',
        'matplotlib',
        
    ]
)
