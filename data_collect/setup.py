import os
from pathlib import Path

from setuptools import find_namespace_packages, setup


PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

with (PROJECT_ROOT / "requirements.txt").open(encoding="utf-8") as requirements_file:
    required = [
        line.strip()
        for line in requirements_file
        if line.strip() and not line.lstrip().startswith("#")
    ]

package_prefixes = ("data_processing",)
packages = find_namespace_packages(
    where=".",
    include=[
        pattern
        for prefix in package_prefixes
        for pattern in (prefix, f"{prefix}.*")
    ],
)


# Include the root project license in standalone distributions.
license_path = Path(__file__).resolve().with_name("LICENSE")
generated_license = not license_path.exists()
if generated_license:
    license_path.write_bytes((license_path.parent.parent / "LICENSE").read_bytes())

try:
    setup(
        name="htd-data",
        version="1.0.0",
        license="MIT",
        packages=packages,
        py_modules=[
            "camera_utils",
            "check_camera",
            "htd_data_collect",
            "tele_vision",
        ],
        description="Humanoid Touch Dream G1 data collection and processing",
        python_requires=">=3.10",
        install_requires=required,
        license_files=["LICENSE"],
    )
finally:
    if generated_license:
        license_path.unlink()
