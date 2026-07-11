"""Environment/package prompt."""

import random


def get_prompt_environment():
    """Installed packages description."""
    pkgs = [
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
    random.shuffle(pkgs)
    pkg_str = ", ".join([f"`{p}`" for p in pkgs])

    return {
        "Installed Packages": (
            f"Prefer the installed packages listed here plus the Python standard library: {pkg_str}. "
            "Do not assume optional packages such as albumentations or tqdm are installed. "
            "If an optional package is not in this list, either avoid it or guard the import with a working fallback. "
            "Do not run pip installs from the solution script. For neural networks we suggest using PyTorch rather than TensorFlow."
        )
    }
