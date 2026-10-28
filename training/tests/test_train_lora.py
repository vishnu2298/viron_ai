import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_lora import require_local_base


def test_rejects_hub_model_ids():
    with pytest.raises(SystemExit, match="Hub models are not allowed"):
        require_local_base("some-org/some-pretrained-model")


def test_rejects_dir_without_config(tmp_path):
    with pytest.raises(SystemExit, match="no config.json"):
        require_local_base(str(tmp_path))


def test_accepts_local_checkpoint_dir(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert require_local_base(str(tmp_path)) == tmp_path
