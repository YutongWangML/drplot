from setuptools import setup

setup(
    name="drplot",
    version="0.1.0",
    author="Yutong Wang",
    description="Decision Region plotting tools",
    packages=["drplot"],
    install_requires=[
        "matplotlib",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
