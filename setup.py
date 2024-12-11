from setuptools import setup, find_packages
import os

long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="vexni",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'ultralytics',
        'transformers',
        'pytesseract',
        'diffusers',
        'torch'
    ],
    entry_points={
        'console_scripts': [
            'vexni=vexni.runner:main',
        ],
    },
    author="Nahom Abera",
    author_email="nahomtesfahun001@gmail.com",
    description="A Domain Specific Language for Computer Vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NahomAbera/Vexni",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)