"""Deterministic symbolic solver and answer verifier."""

from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)
ALLOWED_PATTERN = re.compile(r"^[0-9a-zA-Z_+\-*/^().=\s]+$")


@dataclass(frozen=True)
class Solution:
    normalized_problem: str
    answer: str
    verification: str


def _expression(text: str) -> sp.Expr:
    if not ALLOWED_PATTERN.fullmatch(text):
        raise ValueError("Unsupported characters in expression.")
    return parse_expr(text, transformations=TRANSFORMS, evaluate=True)


def solve_problem(problem: str) -> Solution:
    cleaned = problem.strip()
    if not cleaned:
        raise ValueError("Problem is empty.")

    if "=" in cleaned:
        left, right = cleaned.split("=", 1)
        lhs, rhs = _expression(left), _expression(right)
        symbols = sorted(lhs.free_symbols | rhs.free_symbols, key=lambda item: item.name)
        if not symbols:
            truth = sp.simplify(lhs - rhs) == 0
            return Solution(cleaned, str(truth), f"{lhs} - ({rhs}) = {sp.simplify(lhs-rhs)}")
        roots = sp.solve(sp.Eq(lhs, rhs), symbols, dict=True)
        checks = []
        for root in roots:
            residual = sp.simplify((lhs - rhs).subs(root))
            checks.append(f"{root}: residual={residual}")
        return Solution(cleaned, str(roots), "; ".join(checks) or "No symbolic root found.")

    value = sp.simplify(_expression(cleaned))
    return Solution(cleaned, str(value), f"Re-evaluated symbolic result: {value}")
