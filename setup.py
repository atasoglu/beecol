from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent.resolve()

setup(
    name="beecol",
    version="0.1.0",
    author="Ahmet Atasoglu",
    author_email="ahmetatasoglu98@gmail.com",
    description="A simple Artificial Bee Colony (ABC) library for Python.",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/atasoglu/beecol",
    license="MIT",
    packages=find_packages(),
    install_requires=["numpy"],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)