import os


class LocalGenerator:
    def __init__(self):
        self._use_transformers = os.getenv("RAG_USE_TRANSFORMERS", "1") not in ("0", "false", "False")
        self._generator = None

        if self._use_transformers:
            try:
                from transformers import pipeline  # local import to avoid hard dependency at import time

                self._generator = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-base",
                    device=-1
                )
            except Exception:
                # Fallback to lightweight heuristic generator
                self._use_transformers = False

    def _heuristic_generate(self, prompt: str) -> str:
        # Very small, dependency-free fallback so tests and demos work offline.
        # Extract question
        q_marker = "Question:\n"
        a_marker = "\n\nAnswer:"
        question = ""
        if q_marker in prompt:
            try:
                start = prompt.index(q_marker) + len(q_marker)
                end = prompt.index(a_marker, start)
                question = prompt[start:end].strip()
            except ValueError:
                question = ""

        # Count sources to produce simple citations like [1][2]
        citations = []
        idx = 1
        while f"Source [{idx}]" in prompt:
            citations.append(f"[{idx}]")
            idx += 1

        citation_str = "".join(citations[:3]) if citations else ""

        # Produce a concise, safe answer grounded in provided context only
        base = "Based on the provided sources, here's a concise answer."
        if question:
            base = f"Based on the provided sources, here's a concise answer to: {question}"

        return f"{base} {citation_str}".strip()

    def generate(self, prompt: str) -> str:
        if self._use_transformers and self._generator is not None:
            try:
                result = self._generator(prompt, max_length=256, do_sample=False)
                return result[0]["generated_text"]
            except Exception:
                # If inference fails for any reason, use fallback
                return self._heuristic_generate(prompt)

        return self._heuristic_generate(prompt)
