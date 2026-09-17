import pytest

from solver import solve_problem


def test_arithmetic():
    assert solve_problem("(3 + 4)^2 / 7").answer == "7"


def test_linear_equation():
    result = solve_problem("2x + 5 = 17")
    assert "6" in result.answer
    assert "residual=0" in result.verification


def test_rejects_non_math_input():
    with pytest.raises(ValueError):
        solve_problem("__import__('os').system('echo unsafe')")
