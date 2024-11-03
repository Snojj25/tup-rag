# setup.py
from setuptools import setup, find_packages

setup(
    name="rag",
    version="0.1",
    package_dir={"": "src"},  # tell setuptools packages are under src
    packages=find_packages(where="src"),
    install_requires=[
        # your dependencies here
    ],
)
