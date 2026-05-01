import ast
import operator as op


ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def safe_eval(expr: str):
    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.BinOp):
            if type(node.op) not in ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed")
            return ALLOWED_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed")
            return ALLOWED_OPERATORS[type(node.op)](_eval(node.operand))

        raise ValueError("Invalid expression")

    node = ast.parse(expr, mode="eval").body
    return _eval(node)
