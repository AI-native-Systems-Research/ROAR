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

# Display names only; the stored values in campaigns.models_used are untouched.
# Keys are the strings as recorded; values follow the paper's model table
# (Appendix "Generator models and their providers"). Names already stored in
# that form (eb1, eb1-preview, eb1-delta-preview, eb1-frontier-preview,
# eb1-pro) need no entry.
MODEL_DISPLAY_NAMES = {
    "azure/gpt-5.4": "GPT-5.4",
    "Azure/gpt-5-mini-2025-08-07": "GPT-5-mini",
    "openai/gpt-oss-20b": "GPT-OSS-20B",
    "moonshotai/Kimi-K2.5": "Kimi-K2.5",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
    "gcp/gemini-3-flash-preview": "Gemini-3-Flash-Preview",
    "qwen3.5:9b": "Qwen-3.5-9B",
}


def normalize_model_name(name: str) -> str:
    """Map a stored model name to its paper display form (provider prefix stripped)."""
    if name in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[name]
    if "/" in name:
        return name.split("/", 1)[1]
    return name


# ---------------------------------------------------------------------------
# Mechanism (algorithm_used) display names
# ---------------------------------------------------------------------------

# Display names only; keys are the strings recorded in campaigns.algorithm_used,
# values are the spellings used in the paper text.
MECHANISM_DISPLAY_NAMES = {
    "best_of_n": "Best-of-N",
    "evox": "EvoX",
    "adaevolve": "AdaEvolve",
    "gepa": "GEPA",
    "openevolve": "OpenEvolve",
    "shinka": "ShinkaEvolve",
}


def display_mechanism(name: str) -> str:
    """Map a stored mechanism name to its paper display form; unknown names pass through."""
    return MECHANISM_DISPLAY_NAMES.get(name, name)
