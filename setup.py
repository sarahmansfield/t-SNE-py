import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tsne-sbmansfield", # Replace with your own username
    version="0.0.1",
    author="Sarah Mansfield, Emre Yurtbay",
    author_email="sarah.b.mansfield@duke.edu",
    description="Implementation of t-Distributed Stochastic Neighbor Embedding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sarahmansfield/t-SNE-py",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)