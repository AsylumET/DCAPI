from setuptools import setup, find_packages

setup(
    name="DCAPI",
    version="0.1.0",
    description="Demeter Core API (DCAPI) - A simple API backend.",
    author="Brainshaw",
    license="AGPL-3.0",
    packages=find_packages(),
    install_requires=[
        "flasgger==0.9.7.1",
        "Flask==3.1.0",
        "flask_babel==4.0.0",
        "Flask_Caching==2.3.0",
        "flask_limiter==3.12",
        "flask_sqlalchemy==3.1.1",
        "SQLAlchemy==2.0.31",
        "Werkzeug==3.1.3"
    ],
    classifiers=[
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",  # adjust if needed
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)
