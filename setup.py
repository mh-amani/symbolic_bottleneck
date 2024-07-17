from setuptools import setup, find_packages

setup(
    name="blocks",
    version="0.1",
    packages=find_packages(),
    description="Example package containing blocks modules",
    author="Your Name",
    author_email="your.email@example.com",
    url="http://www.example.com",
    install_requires=[
        # dependencies, if any, e.g., 'numpy >= 1.13.3'
    ],
    python_requires='>=3.6',
)