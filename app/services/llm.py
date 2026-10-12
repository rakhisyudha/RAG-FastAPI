"""LLM client for answer generation.

The client speaks the OpenAI ``/chat/completions`` protocol, which is a de-facto
standard supported by OpenAI, Anthropic-compatible gateways, vLLM, LM Studio
and Ollama. Pointing ``LLM_BASE_URL`` at a different host is therefore all it
takes to switch provider - no code changes.
"""

import httpx

from app.core.config import get_settings

SYSTEM_PROMPT = """You are a precise question-answering assistant.

Rules:
1. Answer using ONLY the information contained in the CONTEXT section.
2. If the context does not contain the answer, reply exactly: \
"I don't have enough information in the provided documents to answer that."
3. Be concise and factual. Do not speculate or add outside knowledge.
4. When you use information from the context, cite it with [1], [2], ... \
matching the numbered sources you were given."""


def build_prompt(question: str, context_blocks: list[str]) -> str:
    """Format the numbered context and the question into a user prompt."""
    if not context_blocks:
        return f"CONTEXT:\n(no relevant documents found)\n\nQUESTION: {question}"

    numbered = "\n\n".join(
        f"[{position}] {block.strip()}" for position, block in enumerate(context_blocks, 1)
    )
    return f"CONTEXT:\n{numbered}\n\nQUESTION: {question}"


def generate(
    question: str,
    context_blocks: list[str],
    temperature: float | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
) -> tuple[str, str]:
    """Generate an answer grounded in the given context.

    Returns:
        A ``(answer, model_name)`` tuple.
    """
    settings = get_settings()
    model = model or settings.llm_model
    user_prompt = build_prompt(question, context_blocks)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": (settings.llm_temperature if temperature is None else temperature),
        "max_tokens": max_tokens or settings.llm_max_tokens,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"

    try:
        # A dedicated client per call keeps the event loop free of shared state;
        # the generous timeout accounts for slow local models.
        with httpx.Client(timeout=settings.llm_timeout_seconds) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Cannot reach the LLM at '{settings.llm_base_url}'. "
            "Start a local server (e.g. `ollama serve`) or set LLM_BASE_URL "
            "to an OpenAI-compatible endpoint in your .env file."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"LLM at '{url}' returned HTTP {exc.response.status_code} "
            f"for model '{model}': {exc.response.text[:300]}"
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"LLM request timed out after {settings.llm_timeout_seconds}s. "
            "Increase LLM_TIMEOUT_SECONDS if the model is slow."
        ) from exc

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM returned no choices: {data}")

    answer = choices[0].get("message", {}).get("content", "").strip()
    if not answer:
        raise RuntimeError("LLM returned an empty answer.")

    return answer, data.get("model", model)
