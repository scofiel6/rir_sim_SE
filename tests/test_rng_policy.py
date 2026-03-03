from tools.check_rng_policy import main as rng_policy_main


def test_rng_policy_passes():
    assert rng_policy_main() == 0
