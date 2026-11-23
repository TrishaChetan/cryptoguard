from cryptoguard.threat_intel import ThreatIntel


def test_lists_load():
    ti = ThreatIntel.load()
    assert 3333 in ti.pool_ports and 443 not in ti.pool_ports
    assert "xmrig" in ti.miner_signatures


def test_matching():
    ti = ThreatIntel.load()
    assert ti.is_pool_port(14444)
    assert not ti.is_pool_port(22)
    assert ti.matches_miner("xmrig") == "xmrig"
    assert ti.matches_miner("python3", "./x -o stratum+tcp://pool:3333") == "stratum+tcp"
    assert ti.matches_miner("nginx", "nginx: worker process") is None
    assert ti.is_blocklisted("203.0.113.7")          # CIDR entry
    assert not ti.is_blocklisted("8.8.8.8")
