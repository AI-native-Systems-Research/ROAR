"""Shared display utilities for analytics figures."""

# ---------------------------------------------------------------------------
# Problem name abbreviations
# ---------------------------------------------------------------------------

PROBLEM_ABBREVIATIONS = {
    "Bounded 2D Knapsack": "2D Knapsack",
    "CUDA Attention kernel optimization": "CUDA Attn",
    "CUDA Layer Norm kernel optimization": "CUDA LN",
    "Polynomino Packing": "Polynomino",
    "Palindrome Hamiltonian Path": "Palindrome",
    "Dispatcher cold path": "Dispatcher",
}


def abbreviate_problem(name: str) -> str:
    """Abbreviate problem name for cleaner axis labels."""
    if name in PROBLEM_ABBREVIATIONS:
        return PROBLEM_ABBREVIATIONS[name]
    for full, abbrev in PROBLEM_ABBREVIATIONS.items():
        if name.startswith(full[:20]) or full.startswith(name[:20]):
            return abbrev
    if len(name) > 15:
        return name[:12] + "..."
    return name


# ---------------------------------------------------------------------------
# Model name normalization
# ---------------------------------------------------------------------------

MODEL_DISPLAY_NAMES = {
    "azure/gpt-5.4": "gpt-5.4",
    "Azure/gpt-5-mini-2025-08-07": "gpt-5-mini",
    "moonshotai/Kimi-K2.5": "Kimi-K2.5",
    "openai/gpt-oss-20b": "gpt-oss-20b",
}


def normalize_model_name(name: str) -> str:
    """Strip provider prefix from model name for figure display."""
    if name in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[name]
    if "/" in name:
        return name.split("/", 1)[1]
    return name
