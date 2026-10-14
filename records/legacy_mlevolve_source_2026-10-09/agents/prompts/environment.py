"""Environment/package prompt."""

import random

from agents.runtime_dependencies import advertised_package_names


def get_prompt_environment(prompt_compatibility_profile: str = "current"):
    """Installed packages description."""
    if (
        str(prompt_compatibility_profile or "current").strip()
        == "original_mlevolve_20260924"
    ):
        packages = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
            "transformers",
            "sentence-transformers",
            "opencv-python",
            "Pillow",
        ]
        random.shuffle(packages)
        package_text = ", ".join(f"`{package}`" for package in packages)
        return {
            "Installed Packages": (
                "Your solution can use any relevant machine learning packages such as: "
                f"{package_text}. Feel free to use any other packages too (all packages "
                "are already installed!). For neural networks we suggest using PyTorch "
                "rather than TensorFlow."
            )
        }

    pkg_str = ", ".join(f"`{package}`" for package in advertised_package_names())

    return {
        "Installed Packages": (
            f"The following packages are guaranteed in the execution environment: {pkg_str}. "
            "Do not assume any unlisted third-party package is installed: either avoid it or "
            "use an ImportError-guarded fallback. Do not run pip from generated code. For neural "
            "networks we suggest using PyTorch rather than TensorFlow."
        )
    }
