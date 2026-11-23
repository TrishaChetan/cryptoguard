from cryptoguard.cli import main


def test_simulate_train_replay(tmp_path, capsys):
    d = tmp_path / "data"
    assert main(["simulate", "--days", "2", "--test-days", "1", "--out-dir", str(d)]) == 0
    m = tmp_path / "m.joblib"
    assert main(["train", "--data", str(d / "normal.csv"), "--calib", str(d / "calib.csv"),
                 "--model", str(m)]) == 0
    log = tmp_path / "alerts.jsonl"
    assert main(["replay", "--data", str(d / "test.csv"), "--model", str(m), "--log", str(log)]) == 0
    out = capsys.readouterr().out
    assert "ALERT" in out and "WHY:" in out
    html = tmp_path / "r.html"
    assert main(["report", "--log", str(log), "--out", str(html)]) == 0
    assert "CryptoGuard incident report" in html.read_text()
