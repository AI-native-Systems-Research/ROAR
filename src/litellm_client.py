"""LiteLLM client for parallelized LLM calls.

Usage:
    from litellm_client import llm_call, llm_batch_call

    # Single call
    result = await llm_call(
        prompt="Summarize this code's approach",
        system_prompt="You are a code analyst.",
    )

    # Batch call (parallelized)
    results = await llm_batch_call(
        prompts=["Summarize code 1...", "Summarize code 2..."],
        system_prompt="You are a code analyst.",
        max_concurrency=10,
    )
"""

import asyncio
import os
from typing import Any, Callable

import litellm
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "aws/claude-haiku-4-5"

litellm.drop_params = True

if os.environ.get("LITELLM_BASE_URL"):
    litellm.api_base = os.environ["LITELLM_BASE_URL"]


async def llm_call(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Make a single LLM call.

    Args:
        prompt: The user prompt
        system_prompt: Optional system prompt
        model: Model to use (defaults to DEFAULT_MODEL)
        temperature: Sampling temperature (default 0.0 for deterministic)
        max_tokens: Maximum tokens in response

    Returns:
        The model's response text
    """
    model = model or os.environ.get("LITELLM_MODEL", DEFAULT_MODEL)

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = await litellm.acompletion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        custom_llm_provider="openai",
    )

    return response.choices[0].message.content


async def _llm_call_with_index(
    index: int,
    prompt: str,
    system_prompt: str | None,
    model: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> tuple[int, str | None]:
    """Helper to make an LLM call with index tracking and concurrency control."""
    async with semaphore:
        try:
            result = await llm_call(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (index, result)
        except Exception as e:
            print(f"Error on prompt {index}: {e}")
            return (index, None)


async def llm_batch_call(
    prompts: list[str],
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_concurrency: int = 10,
    progress_callback: Callable[[int, int], None] | None = None,
    result_callback: Callable[[int, str | None], None] | None = None,
) -> list[str | None]:
    """Make parallelized LLM calls for a batch of prompts.

    Args:
        prompts: List of user prompts
        system_prompt: Optional system prompt (shared across all calls)
        model: Model to use (defaults to DEFAULT_MODEL)
        temperature: Sampling temperature (default 0.0 for deterministic)
        max_tokens: Maximum tokens in response
        max_concurrency: Maximum concurrent requests (default 10)
        progress_callback: Optional callback(completed, total) for progress updates
        result_callback: Optional callback(index, result) called as each result arrives

    Returns:
        List of response texts in the same order as prompts (None for failures)
    """
    model = model or os.environ.get("LITELLM_MODEL", DEFAULT_MODEL)
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        _llm_call_with_index(
            index=i,
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            semaphore=semaphore,
        )
        for i, prompt in enumerate(prompts)
    ]

    results: list[str | None] = [None] * len(prompts)
    completed = 0

    for coro in asyncio.as_completed(tasks):
        index, result = await coro
        results[index] = result
        completed += 1
        if result_callback:
            result_callback(index, result)
        if progress_callback:
            progress_callback(completed, len(prompts))

    return results


def llm_call_sync(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Synchronous wrapper for llm_call."""
    return asyncio.run(
        llm_call(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )


def llm_batch_call_sync(
    prompts: list[str],
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_concurrency: int = 10,
    progress_callback: Callable[[int, int], None] | None = None,
    result_callback: Callable[[int, str | None], None] | None = None,
) -> list[str | None]:
    """Synchronous wrapper for llm_batch_call."""
    return asyncio.run(
        llm_batch_call(
            prompts=prompts,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrency=max_concurrency,
            progress_callback=progress_callback,
            result_callback=result_callback,
        )
    )
