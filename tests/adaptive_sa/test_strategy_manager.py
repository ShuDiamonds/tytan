from tytan.adaptive_sa import StrategyManager


def _build_configs():
    return [
        {"name": "foo", "type": "linear", "weight": 0.5},
        {"name": "bar", "type": "exponential", "weight": 1.0},
    ]


def test_strategy_manager_prefers_highest_weight_when_epsilon_zero():
    manager = StrategyManager(strategies=_build_configs(), epsilon=0.0, seed=0)
    selection = manager.select()
    assert selection["name"] == "bar"
    manager.record("bar", 0.4)
    weights = manager.weights
    assert weights["bar"] > weights["foo"]


def test_strategy_manager_explores_when_epsilon_one():
    manager = StrategyManager(strategies=_build_configs(), epsilon=1.0, seed=1)
    names = {manager.select()["name"] for _ in range(5)}
    assert "foo" in names
    assert "bar" in names
