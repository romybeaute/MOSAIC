from setuptools import setup, find_packages

setup(
    name="MOSAIC",
    packages=find_packages() + ['configs'],
    package_dir={'configs': 'configs'},
    version="0.1",
    author="Romy Beauté",
    author_email="r.beaut@sussex.ac.uk"
)