import json
import types
import sys

import pytest
from tytan import symbols, Compile
from tytan.sampler import NQSLocalSampler


@pytest.fixture
def fake_httpx(monkeypatch):
    text = json.dumps({"energy": 0.0, "result": {"x": 0, "y": 0, "z": 0}, "time": 0.0})
    response = types.SimpleNamespace(text=text)

    def post(*args, **kwargs):
        return response

    module = types.ModuleType("httpx")
    module.post = post
    monkeypatch.setitem(sys.modules, "httpx", module)
    return module


def test_nqs_local_sampler_run(fake_httpx):
    x, y, z = symbols("x y z")
    expr = 3 * x**2 + 2 * x * y + 4 * y**2 + z**2 + 2 * x * z + 2 * y * z
    qubo, offset = Compile(expr).get_qubo()
    sampler = NQSLocalSampler()
    result = sampler.run(qubo)
    assert result is not None
    assert result[0][0] is not None
    assert result[0][0]["x"] == 0
    assert result[0][0]["y"] == 0
    assert result[0][0]["z"] == 0
    assert result[0][1] == 0  # energy
    assert result[0][2] is not None  # occ
    assert result[0][3] is not None  # time
