from __future__ import annotations

from agents.prompts.validation_template_prompts import get_code_review_guidelines


def test_code_review_allows_trained_fresh_initialization_in_offline_runs() -> None:
    """Offline review must not confuse fresh initialization with dummy inference."""

    text = "\n".join(get_code_review_guidelines())

    assert "freshly initialized trainable model is valid" in text.lower()
    assert "do not flag weights=none or pretrained=false" in text.lower()
    assert "all models (including those released after your training data cutoff) are available" not in text.lower()


def test_shape_findings_require_an_executable_dimension_contradiction() -> None:
    text = "\n".join(get_code_review_guidelines()).lower()

    assert "proves that the tensor constructed at the model call" in text
    assert "deliberate, consistently applied feature subset is not a shape failure" in text


def test_review_requires_imports_before_definition_time_torch_references() -> None:
    text = "\n".join(get_code_review_guidelines()).lower()

    assert "import ordering" in text
    assert "torch.tensor" in text


def test_review_requires_worker_compatible_sklearn_rmse() -> None:
    text = "\n".join(get_code_review_guidelines()).lower()

    assert "np.sqrt(mean_squared_error" in text
    assert "squared=false" in text
