import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryptoguard.model import BaselineModel  # noqa: E402
from cryptoguard.simulator import generate_normal, generate_test  # noqa: E402


@pytest.fixture(scope="session")
def data():
    normal = generate_normal(days=3, seed=11)
    calib = generate_normal(days=1, seed=12, start_day=3)
    test, events = generate_test(days=2, seed=13, start_day=4, n_noisy=2, n_stealthy=1,
                                 n_container=1, n_surge=1, n_ci=1)
    return normal, calib, test, events


@pytest.fixture(scope="session")
def model(data):
    normal, calib, _, _ = data
    return BaselineModel(n_estimators=100, random_state=0).fit(normal, calib)


@pytest.fixture()
def cfg(tmp_path):
    from cryptoguard.config import load_config
    c = load_config()
    c["response"]["notify"]["log_file"] = str(tmp_path / "alerts.jsonl")
    return c
