import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dabstract", # Replace with your own username
    version="0.0.1",
    author="Gert Dekkers, DTAI_ADVISE, KU Leuven",
    author_email="gert.dekkers@gmail.com",
    description="Database and preprocessing abstraction library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KULeuvenADVISE/dabstract",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)