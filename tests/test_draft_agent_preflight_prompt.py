"""Regression coverage for generated-candidate preflight requirements."""

from agents.draft_agent import model_preflight_generation_instructions


def test_generation_instructions_require_complete_candidate_adapter() -> None:
    """A draft must be told the exact interface required before GPU admission."""
    instructions = "\n".join(model_preflight_generation_instructions())

    assert "class CandidateAdapter" in instructions
    for method in (
        "build_model",
        "build_optimizer",
        "build_train_batch",
        "build_validation_batch",
        "training_step",
        "validation_step",
    ):
        assert method in instructions
    assert "if __name__ == '__main__'" in instructions
