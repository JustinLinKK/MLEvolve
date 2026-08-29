"""Environment/package prompt."""

from agents.runtime_dependencies import advertised_package_names


def get_prompt_environment():
    """Installed packages description."""
    pkg_str = ", ".join(f"`{package}`" for package in advertised_package_names())

    return {
        "Installed Packages": (
            f"The following packages are guaranteed in the execution environment: {pkg_str}. "
            "Do not assume any unlisted third-party package is installed: either avoid it or "
            "use an ImportError-guarded fallback. Do not run pip from generated code. For neural "
            "networks we suggest using PyTorch rather than TensorFlow."
        )
    }
