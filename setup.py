# ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD!

from distutils.core import setup
from setuptools import find_packages

setup(
    name="vgn",
    version='0.1.0',
    packages=(find_packages(include=["vgn","vgn.*"])),
    package_dir={"": "src"}
    )
