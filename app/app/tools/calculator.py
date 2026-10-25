"""Tool: safe arithmetic calculator (AST-based; never uses eval())."""
from __future__ import annotations

import ast
import math
import operator

from ..guardrails import ToolInputError

NAME = "calculator"

DESCRIPTION = (
    "Safe arithmetic calculator for percentages, ratios, sums, savings projections or SIP maths — anything that "
    "is NOT an EMI (use calculate_emi for loans). Supports + - * / ** // % ( ) and round, abs, min, max, sqrt. "
    "Write percentages as fractions, e.g. 18% of 85000 -> '0.18 * 85000'. Example: '85000 * 12 * 0.3'."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string", "maxLength": 200,
            "description": "Arithmetic expression using digits and operators only, e.g. '(85000 - 42000) / 85000 * 100'.",
        },
    },
    "required": ["expression"],
}
MAX_CALLS = 6

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max, "sqrt": math.sqrt}


def _ev(node, depth=0):
    if depth > 25:
        raise ToolInputError("Expression is too deeply nested.")
    if isinstance(node, ast.Expression):
        return _ev(node.body, depth + 1)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _ev(node.left, depth + 1), _ev(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ToolInputError("Exponent too large (max 100).")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_ev(node.operand, depth + 1))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
            and not node.keywords and 1 <= len(node.args) <= 5):
        return _FUNCS[node.func.id](*[_ev(a, depth + 1) for a in node.args])
    raise ToolInputError("Only numbers, + - * / ** // % ( ) and round/abs/min/max/sqrt are allowed.")


def validate(args: dict, ctx) -> dict:
    expr = args["expression"].replace("^", "**").replace("×", "*").replace("÷", "/").replace("₹", "")
    expr = expr.replace("_", "")
    if expr.rstrip().endswith("%"):
        raise ToolInputError("Write percentages as fractions, e.g. 18% -> 0.18.")
    args["expression"] = expr
    return args


def run(args: dict, ctx=None) -> dict:
    expr = args["expression"]
    try:
        tree = ast.parse(expr, mode="eval")
        value = _ev(tree)
    except ToolInputError:
        raise
    except ZeroDivisionError:
        raise ToolInputError("Division by zero.")
    except (SyntaxError, ValueError, TypeError, OverflowError) as e:
        raise ToolInputError(f"Could not evaluate expression: {e}")
    if isinstance(value, float):
        value = round(value, 6)
    return {"expression": expr, "result": value, "summary_line": f"{expr} = {value}"}
