#!/usr/bin/env python3
"""Diagnose LLM access for refresh_literature_db.py's filter classification.

Sends the classification request step by step, in the same setting as the refresh
script (same config, model, prompt, a real document from a build, threads), and
reports which step fails. Prints no secrets (only the key's length and last 4 chars).

    python db_initializations/check_llm_access.py --out-dir chroma_data_YYYYMMDD
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
KEY_FROM_SHELL = "OPENAI_API_KEY" in os.environ
import config  # noqa: E402
import refresh_literature_db as refresh  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", help="a build directory: test with real documents from its collection")
    p.add_argument("--llm-model", default=config.LLM_MODEL)
    p.add_argument("--collection", default=config.CHROMA_COLLECTION)
    args = p.parse_args()

    import httpx
    import openai
    from openai import OpenAI

    key = (config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY") or "").strip()
    print(f"python:   {sys.executable}")
    print(f"openai:   {openai.__version__} | httpx: {httpx.__version__}")
    print(f"base URL: {config.OPENAI_BASE_URL!r}")
    print(f"model:    {args.llm_model!r}   (repr: hidden characters would show here)")
    print(f"api key:  {len(key)} chars, ...{key[-4:] if key else ''}, "
          f"{'from the shell environment' if KEY_FROM_SHELL else 'from evidence-database/.env'}")
    print(f"OPENAI_* variables set: {sorted(v for v in os.environ if v.startswith('OPENAI'))}")
    print(f"proxy variables set:    {sorted(v for v in os.environ if 'proxy' in v.lower())}\n")

    groups = refresh._filter_schema()
    groups = groups[0] if isinstance(groups, tuple) else groups
    docs = ["Enzalutamide in metastatic castration-resistant prostate cancer: a phase III randomised trial. "
            "We randomly assigned 1199 men with mCRPC after docetaxel to enzalutamide or placebo."]
    if args.out_dir:
        import chromadb
        col = chromadb.PersistentClient(path=str(args.out_dir)).get_collection(args.collection)
        got = col.get(limit=5, include=["documents"])
        docs = [d for d in got["documents"] if d] or docs
        print(f"(using {len(docs)} real documents from {args.out_dir}; chromadb imported like the refresh script)\n")

    client = OpenAI(api_key=key, base_url=config.OPENAI_BASE_URL, max_retries=0, timeout=120)

    def ask(label, messages, **kw):
        try:
            resp = client.chat.completions.create(model=args.llm_model, messages=messages, **kw)
            out = f"OK ({(resp.choices[0].message.content or '')[:60]!r})"
        except Exception as exc:  # noqa: BLE001
            body = str(getattr(exc, "body", "") or exc)[:200]
            if len(key) >= 8:
                body = body.replace(key, key[:2] + "…" + key[-4:])
            out = f"FAILED {type(exc).__name__} {getattr(exc, 'status_code', '')} {body}"
        print(f"  {label}: {out}")
        return out.startswith("OK")

    def script_messages(doc):
        return [{"role": "system", "content": refresh._EXTRACT_SYSTEM},
                {"role": "user", "content": refresh._extract_prompt(doc, groups)}]

    print("Requests:")
    ask("1 short message", [{"role": "user", "content": "hi"}], max_tokens=5)
    ask("2 script prompt, no JSON mode", script_messages(docs[0]), temperature=0)
    ask("3 script request (JSON mode, temperature 0)", script_messages(docs[0]), temperature=0,
        response_format={"type": "json_object"})
    for i, doc in enumerate(docs[1:], start=2):
        ask(f"3.{i} script request, document {i}", script_messages(doc), temperature=0,
            response_format={"type": "json_object"})
    print("  4 script requests in 2 threads:")
    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(lambda d: ask("    thread", script_messages(d), temperature=0,
                                  response_format={"type": "json_object"}), docs[:4]))
    print("\n  5 the refresh script's own function:")
    result = refresh._classify_one(client, args.llm_model, docs[0], groups, retries=1)
    print(f"    _classify_one -> {'OK' if result is not None else 'FAILED (see message above)'}")


if __name__ == "__main__":
    main()
