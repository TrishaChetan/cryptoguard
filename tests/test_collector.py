from cryptoguard.collector import TelemetryCollector
from cryptoguard.features import BEHAVIOURAL_FEATURES, INTEL_FEATURES
from cryptoguard.threat_intel import ThreatIntel


def test_collect_window_has_all_features():
    col = TelemetryCollector(ThreatIntel.load(), window_seconds=1, sample_seconds=0.5)
    w = col.collect_window()
    for f in BEHAVIOURAL_FEATURES + INTEL_FEATURES:
        assert f in w.features
    assert 0 <= w.features["cpu_percent"] <= 100
    assert w.features["num_procs"] > 0
