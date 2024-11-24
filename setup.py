from setuptools import setup, find_packages

setup(
    name="pyfi",  # Your package name
    version="0.1.0",  # Initial version
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_package",  # Replace with your repo URL
    packages=find_packages(),  # Automatically discover all packages and subpackages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version required
    install_requires=[
        # List your package dependencies here
        "requests",
        "numpy",
    ],
    include_package_data=True,  # Include files specified in MANIFEST.in
)
