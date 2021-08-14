import setuptools

with open("README.md", "r", encoding="utf-8") as rme:
    long_description = rme.read()

with open("requirements.txt", "r", encoding="utf-8") as req:
    requirements = req.read()

setuptools.setup(
    name="bb8TSA",
    version="0.1.0",
    author="Farhang Habibi",
    author_email="farhang.habibi@bpifrance.fr",
    description="Extract some statistical indicators from time series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.ialab-bpifrance.fr/data-ia/bpi-fr-data-sc-bb8-ts-analysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=requirements
)