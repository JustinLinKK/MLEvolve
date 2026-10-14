import os
import random


def set_global_seed(seed: int) -> None:
    """Set random seeds across common libraries for reproducibility."""

    random.seed(seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)

    except Exception:
        pass

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    except Exception:
        pass
