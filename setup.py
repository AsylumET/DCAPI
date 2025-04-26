from setuptools import setup, find_packages

setup(
    name="DCAPI",
    version="0.1.0",
    description="Demeter Core API (DCAPI) - A simple API backend.",
    author="Brainshaw",
    license="AGPL-3.0",
    packages=find_packages(),
    install_requires=[
        "Flask",
        "requests",
    ],
    classifiers=[
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",  # adjust if needed
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)
