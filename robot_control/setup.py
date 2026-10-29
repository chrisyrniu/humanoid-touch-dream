from pathlib import Path

from setuptools import find_packages
from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()


# Include the root project license in standalone distributions.
license_path = Path(__file__).resolve().with_name("LICENSE")
generated_license = not license_path.exists()
if generated_license:
    license_path.write_bytes((license_path.parent.parent / "LICENSE").read_bytes())

try:
    setup(
        name='htd-robot-control',
        version='1.0.0',
        license="MIT AND BSD-3-Clause",
        packages=find_packages(),
        description='Humanoid Touch Dream G1 robot controller',
        install_requires=required,
        license_files=["LICENSE", "licenses/*.txt"],
    )
finally:
    if generated_license:
        license_path.unlink()
