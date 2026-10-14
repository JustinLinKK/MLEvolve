from pathlib import Path

from engine.executor import _build_perfseer_model_metadata, _count_labeled_training_rows


def test_perfseer_metadata_includes_num_tabular_features_input(tmp_path: Path) -> None:
    """A multimodal candidate must expose its image and tabular inputs."""
    metadata = _build_perfseer_model_metadata(
        """
NUM_TABULAR_FEATURES = 12

def build_model(context=None):
    return object()
""",
        source_path=tmp_path / "candidate.py",
        input_resolution=256,
        precision_mode="fp32",
        dataset_size=3,
    )

    assert metadata is not None
    assert metadata["input_shapes"] == [["$batch", 3, 256, 256], ["$batch", 12]]
    assert metadata["input_dtypes"] == ["float32", "float32"]
    assert metadata["dataset_size"] == 3


def test_perfseer_metadata_includes_num_tab_features_input(tmp_path: Path) -> None:
    """The PetFinder agent commonly names the fusion width ``NUM_TAB_FEATURES``."""
    metadata = _build_perfseer_model_metadata(
        """
NUM_TAB_FEATURES = 17

def build_model(context=None):
    return object()
""",
        source_path=tmp_path / "candidate.py",
        input_resolution=224,
        precision_mode="fp32",
    )

    assert metadata is not None
    assert metadata["input_shapes"] == [["$batch", 3, 224, 224], ["$batch", 17]]


def test_perfseer_metadata_resolves_static_tabular_width_expression(tmp_path: Path) -> None:
    """PetFinder candidates derive fusion width from a constant feature list."""
    metadata = _build_perfseer_model_metadata(
        """
FLAG_COLS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
NUM_TAB_FEATURES = len(FLAG_COLS) + 5

def build_model(context=None):
    return object()
""",
        source_path=tmp_path / "candidate.py",
        input_resolution=224,
        precision_mode="fp32",
    )

    assert metadata is not None
    assert metadata["input_shapes"] == [["$batch", 3, 224, 224], ["$batch", 17]]


def test_perfseer_metadata_recognizes_petfinder_features_alias(tmp_path: Path) -> None:
    """PetFinder's common ``FEATURES`` list is a tabular model input."""
    metadata = _build_perfseer_model_metadata(
        '''
FEATURES = ["a", "b", "c"]

def build_model(context=None):
    return object()
''',
        source_path=tmp_path / "candidate.py",
        input_resolution=224,
        precision_mode="fp32",
    )

    assert metadata is not None
    assert metadata["input_shapes"] == [["$batch", 3, 224, 224], ["$batch", 3]]


def test_perfseer_metadata_wraps_single_noarg_module_class(tmp_path: Path) -> None:
    """Generated candidates often expose one model class instead of ``build_model``."""
    source_path = tmp_path / "candidate.py"
    metadata = _build_perfseer_model_metadata(
        """
import torch.nn as nn

TABULAR_COLS = ["a", "b", "c"]

class DualPathModel(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.dropout = dropout

    def forward(self, images, tabular):
        return images
""",
        source_path=source_path,
        input_resolution=256,
        precision_mode="fp32",
    )

    assert metadata is not None
    assert metadata["source_path"] == str(tmp_path / "candidate_perfseer_v32.py")
    assert metadata["input_shapes"] == [["$batch", 3, 256, 256], ["$batch", 3]]
    assert "return DualPathModel()" in (tmp_path / "candidate_perfseer_v32.py").read_text()


def test_perfseer_metadata_wraps_explicit_root_module_among_dataset_classes(
    tmp_path: Path,
) -> None:
    """A top-level trained model assignment disambiguates generated helpers."""
    source_path = tmp_path / "candidate.py"
    metadata = _build_perfseer_model_metadata(
        """
import torch.nn as nn

class PetDataset:
    def __init__(self, dataframe):
        self.dataframe = dataframe

class FeatureBlock(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, images):
        return images

class PawpularityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = FeatureBlock()
    def forward(self, images):
        return self.features(images)

model = PawpularityModel()
""",
        source_path=source_path,
        input_resolution=224,
        precision_mode="fp32",
    )

    assert metadata is not None
    assert "return PawpularityModel()" in (
        tmp_path / "candidate_perfseer_v32.py"
    ).read_text()


def test_perfseer_metadata_uses_constructor_default_when_assignment_is_dynamic(
    tmp_path: Path,
) -> None:
    """A runtime-derived constructor value must not leak into static capture."""
    source_path = tmp_path / "candidate.py"
    metadata = _build_perfseer_model_metadata(
        '''
import torch.nn as nn

class PawpularityModel(nn.Module):
    def __init__(self, num_meta_features=30):
        super().__init__()
    def forward(self, images, meta):
        return images

num_meta_features = dataframe.shape[1]
model = PawpularityModel(num_meta_features=num_meta_features)
''',
        source_path=source_path,
        input_resolution=224,
        precision_mode="fp32",
    )

    assert metadata is not None
    captured = (tmp_path / "candidate_perfseer_v32.py").read_text()
    assert "return PawpularityModel()" in captured
    assert "num_meta_features = dataframe.shape[1]" not in captured


def test_perfseer_metadata_wraps_static_timm_feature_extractor(
    tmp_path: Path,
) -> None:
    """A hybrid feature-extraction candidate exposes its real GPU backbone."""
    source_path = tmp_path / "candidate.py"
    metadata = _build_perfseer_model_metadata(
        """
import timm
MODEL_NAME = "vit_base_patch16_224"
feature_extractor = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
""",
        source_path=source_path,
        input_resolution=224,
        precision_mode="fp32",
    )

    assert metadata is not None
    assert "return timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)" in (
        tmp_path / "candidate_perfseer_v32.py"
    ).read_text()


def test_perfseer_capture_never_downloads_timm_pretrained_weights(tmp_path: Path) -> None:
    """Static topology capture must not block on a model-hub weight download."""
    source_path = tmp_path / "candidate.py"
    metadata = _build_perfseer_model_metadata(
        '''
import torch.nn as nn
import timm

class CandidateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=True, num_classes=0)
    def forward(self, images):
        return self.backbone(images)

model = CandidateModel()
''',
        source_path=source_path,
        input_resolution=224,
        precision_mode="fp32",
    )

    assert metadata is not None
    captured = (tmp_path / "candidate_perfseer_v32.py").read_text()
    assert "pretrained=False" in captured
    assert "pretrained=True" not in captured


def test_perfseer_metadata_reads_tabular_width_from_candidate_adapter(tmp_path: Path) -> None:
    """PetFinder adapters may keep their multimodal width on ``self`` only."""
    metadata = _build_perfseer_model_metadata(
        """
import torch.nn as nn

class CandidateAdapter:
    def __init__(self):
        self.tabular_dim = 12
    def build_model(self, context):
        return PetFinderModel(tabular_dim=self.tabular_dim)

class PetFinderModel(nn.Module):
    def __init__(self, tabular_dim=12):
        super().__init__()
    def forward(self, images, tabular):
        return images
""",
        source_path=tmp_path / "candidate.py",
        input_resolution=224,
        precision_mode="fp32",
    )

    assert metadata is not None
    assert metadata["input_shapes"] == [["$batch", 3, 224, 224], ["$batch", 12]]


def test_perfseer_metadata_reads_historical_petfinder_feature_cols(tmp_path: Path) -> None:
    """The frozen 9/2 ViT candidate names its tabular inputs ``FEATURE_COLS``."""
    metadata = _build_perfseer_model_metadata(
        """
import torch.nn as nn

FEATURE_COLS = ["a", "b", "c", "d"]

class PawpularityModel(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
    def forward(self, images, meta):
        return images
""",
        source_path=tmp_path / "candidate.py",
        input_resolution=384,
        precision_mode="fp32",
    )

    assert metadata is not None
    assert metadata["input_shapes"] == [["$batch", 3, 384, 384], ["$batch", 4]]


def test_counts_labeled_training_rows_from_petfinder_csv(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "Id,Pawpularity\na,10\nb,20\nc,30\n",
        encoding="utf-8",
    )

    assert _count_labeled_training_rows(train_csv) == 3
