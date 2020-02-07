import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="owlmPy",
    version="0.0.7",
    author="Oscar Munoz, Nivolas Avilan, Cesar Herreno",
    author_email="oscar.munozs@utadeo.edu.co",
    description="A python package for the modeling and simulation of the propagation of electromagnetic waves in structured multilayers with magneto-optical activity using FDTD method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mojv/owlmPy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)