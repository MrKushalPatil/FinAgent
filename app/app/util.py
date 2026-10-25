"""Small formatting helpers."""


def inr(x: float, decimals: int = 0) -> str:
    """Format a number with Indian digit grouping: 5000000 -> ₹50,00,000."""
    neg = x < 0
    s = f"{abs(x):.{decimals}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    out = "₹" + whole + (("." + frac) if decimals else "")
    return "-" + out if neg else out
