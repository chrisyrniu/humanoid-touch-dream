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

package_prefixes = ("detr", "configs")
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
        name="htd-policy",
        version="1.0.0",
        license="MIT AND Apache-2.0",
        packages=packages,
        py_modules=[
            "mxt_utils",
            "policy",
            "train_act",
            "train_mxt",
            "act_utils",
        ],
        package_data={
            package: ["*.yaml", "*.sh", "*.txt", "LICENSE", "README.md"]
            for package in packages
        },
        description="Humanoid Touch Dream policy training",
        python_requires=">=3.10",
        install_requires=required,
        license_files=["LICENSE", "detr/LICENSE", "detr/licenses/*.txt"],
    )
finally:
    if generated_license:
        license_path.unlink()
