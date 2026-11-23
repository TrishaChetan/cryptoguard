import numpy as np

from cryptoguard.model import BaselineModel


def test_threshold_and_scores(model, data):
    _, calib, test, _ = data
    cal = model.score(calib)
    assert abs((cal > model.threshold).mean() - 0.005) < 0.01
    noisy = test[test["scenario"] == "noisy"]
    normal = test[test["scenario"] == "normal"]
    assert (model.score(noisy) > model.threshold).mean() > 0.9
    assert (model.score(normal) > model.threshold).mean() < 0.05


def test_cpu_excess_is_one_sided(model, data):
    _, _, test, _ = data
    d = model.with_derived(test)
    assert (d["cpu_excess"] >= 0).all()
    assert d.loc[test["scenario"] == "stealthy", "cpu_excess"].median() > \
           d.loc[test["scenario"] == "normal", "cpu_excess"].quantile(0.95)


def test_confidence_mapping(model):
    c = model.behavioural_confidence(np.array([model.threshold]))
    assert abs(c[0] - 0.5) < 1e-9


def test_save_load(model, data, tmp_path):
    _, _, test, _ = data
    p = tmp_path / "m.joblib"
    model.save(p)
    m2 = BaselineModel.load(p)
    assert np.allclose(m2.score(test.head(50)), model.score(test.head(50)))
