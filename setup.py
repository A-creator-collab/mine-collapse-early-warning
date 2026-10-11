from setuptools import setup, find_packages

setup(
    name="mine-collapse-early-warning",
    version="1.0.0",
    author="Your Name",
    description="AI-powered mine collapse early-warning system (software prototype)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=open("requirements.txt").read().splitlines(),
)