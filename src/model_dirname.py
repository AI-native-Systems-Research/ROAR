"""Locate model-family tokens in a run's directory name, returned VERBATIM.

Last-resort fallback for parsers whose systems do not record the model in their
structured output. This module only *finds and returns* the literal substring that
is already present in the path — it performs NO normalization or canonical mapping
(it returns e.g. ``gpt5``, not ``gpt-5``; ``gemini3pro``, not ``gemini-3-pro-preview``).
"""

from pathlib import Path

# Known LLM family prefixes (lowercased). A path segment whose lowercased form starts
# with one of these is treated as a model token and returned exactly as written.
_FAMILIES = (
    "claude", "gpt", "gemini", "deepseek", "qwen", "kimi",
    "llama", "mistral", "grok", "phi", "o1", "o3",
)


def _tokens_in_name(name: str) -> list[str]:
    """Model tokens found in a single dir name, verbatim, in order of appearance."""
    # Primary: underscore-delimited segments — models keep their own hyphens, e.g.
    # `claude-sonnet-4-6`, `gemini-3-pro-preview`, `gpt5`, `gemini3pro`.
    found = [seg for seg in name.split("_") if seg.lower().startswith(_FAMILIES)]
    if found:
        return found
    # Fallback: hyphen-delimited sub-parts (e.g. a parent dir `llm-sql-gemini`).
    return [p for p in name.replace("_", "-").split("-") if p.lower().startswith(_FAMILIES)]


def models_from_path(path, max_levels: int = 3) -> list[str] | None:
    """Verbatim model tokens from the run dir name, walking up to `max_levels` parents.

    Returns the tokens from the closest ancestor that has any (de-duplicated, order
    preserved), or None if none of the inspected names contain a model token.
    """
    p = Path(path)
    for _ in range(max_levels):
        toks = _tokens_in_name(p.name)
        if toks:
            seen: set[str] = set()
            return [t for t in toks if not (t in seen or seen.add(t))]
        if p.parent == p:
            break
        p = p.parent
    return None
