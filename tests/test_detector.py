import numpy as np

from cryptoguard.detector import Detector
from cryptoguard.explainer import Explainer


def test_streaming_equals_batch(model, data):
    _, _, test, _ = data
    exp = Explainer(model)
    part = test.iloc[: 400]
    batch = Detector(model, explainer=exp).run(part)
    det = Detector(model, explainer=exp)
    stream = [det.process(part.iloc[[i]]).confidence for i in range(len(part))]
    assert np.allclose(batch["confidence"].to_numpy(), stream)


def test_intel_alone_raises_alert(model, data):
    _, _, test, _ = data
    row = test[test["scenario"] == "normal"].iloc[[0]].copy()
    row["pool_port_conns"] = 2
    v = Detector(model).process(row)
    assert v.alert and v.intel_conf > 0


def test_every_attack_detected_no_hard_negative_alert(model, data):
    _, _, test, events = data
    v = Detector(model, explainer=Explainer(model)).run(test)
    for e in events:
        seg = v["alert"].iloc[e.start: e.start + e.duration]
        if e.malicious:
            assert seg.any(), f"missed {e.kind}"
    surge = test["scenario"] == "traffic_surge"
    assert v.loc[surge, "alert"].mean() < 0.05
