"""Retrieval (BM25, pure Python) and answering (AI if a key is set, otherwise extractive)."""
from __future__ import annotations

import json
import math
import re
import urllib.request
from collections import Counter

STOPWORDS = set("""a an the and or of to in on for with is are was were be been it this that these those as at by from
what which who whom how why when where do does did can could should would will about into than then there their
they them you your i we our not no yes if so but""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25Index:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c["text"])) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df = Counter(t for d in self.docs for t in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 4) -> list[tuple[float, dict]]:
        q = tokenize(query)
        scored = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            for t in q:
                f = doc.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1))
                    score += self.idf[t] * f * (self.k1 + 1) / norm
            if score > 0:
                scored.append((score, self.chunks[i]))
        return sorted(scored, key=lambda x: -x[0])[:k]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 4]


def answer_extractive(question: str, hits: list[tuple[float, dict]], n: int = 3) -> str:
    """No AI: pick the sentences that best match the question."""
    if not hits:
        return "I couldn't find anything about that in the uploaded PDFs."
    q = set(tokenize(question))
    ranked = []
    for _, chunk in hits:
        for s in _sentences(chunk["text"]):
            overlap = len(q & set(tokenize(s)))
            if overlap:
                ranked.append((overlap, s, chunk))
    ranked.sort(key=lambda x: -x[0])
    seen, lines = set(), []
    for _, s, c in ranked:
        if s not in seen:
            seen.add(s)
            lines.append(f"- {s} *(p. {c['page']}, {c['source']})*")
        if len(lines) == n:
            break
    return "\n".join(lines) or "I found related passages but no sentence that answers it directly. See the sources below."


def answer_with_ai(question: str, hits: list[tuple[float, dict]], api_key: str,
                   model: str = "claude-sonnet-5-5") -> str:
    """Ask Claude to answer ONLY from the retrieved passages (needs an Anthropic API key)."""
    context = "\n\n".join(f"[{c['source']} p.{c['page']}]\n{c['text']}" for _, c in hits)
    prompt = ("Answer the question using ONLY the excerpts below. Cite sources like [file p.N]. "
              "If the answer is not in the excerpts, say you could not find it.\n\n"
              f"Excerpts:\n{context}\n\nQuestion: {question}")
    body = json.dumps({"model": model, "max_tokens": 700,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST", headers={
        "content-type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return "".join(b.get("text", "") for b in data.get("content", [])).strip()


def answer(question: str, hits: list[tuple[float, dict]], api_key: str = "") -> tuple[str, str]:
    """Return (answer_text, mode). Falls back to extractive if the AI call fails."""
    if api_key and hits:
        try:
            return answer_with_ai(question, hits, api_key), "AI"
        except Exception as err:
            return answer_extractive(question, hits) + f"\n\n*(AI unavailable: {err}. Showing best matches instead.)*", "extractive"
    return answer_extractive(question, hits), "extractive"
