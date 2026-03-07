#!/usr/bin/env python3
"""
Evaluate `full_context` versus `pageindex` answer quality across models.

This runner uses the OpenAI Responses API for:
- PageIndex tree search
- Answer generation
- Reference-guided judging

It reads tracked `*_structure.json` files, enriches them with page text at runtime,
and writes Markdown plus JSON reports for local runs or GitHub Actions.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pageindex_core as core  # noqa: E402

DEFAULT_DATASET_PATH = REPO_ROOT / "evals" / "pageindex_vs_full_context.yaml"
DEFAULT_MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]
DEFAULT_JUDGE_MODEL = "gpt-4.1"
DEFAULT_TEXT_VERBOSITY = "medium"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".dist" / "evals"
OPENAI_EVAL_FLYWHEEL_URL = (
    "https://developers.openai.com/cookbook/examples/evaluation/"
    "building_resilient_prompts_using_an_evaluation_flywheel/"
)
OPENAI_OPTIMIZING_ACCURACY_URL = (
    "https://developers.openai.com/api/docs/guides/optimizing-llm-accuracy/"
)
TREE_TOP_K = 5
TREE_SEARCH_MAX_OUTPUT_TOKENS = 300
ANSWER_MAX_OUTPUT_TOKENS = 500
JUDGE_MAX_OUTPUT_TOKENS = 700
CITATION_LEVELS = {"missing", "weak", "adequate", "strong"}

TREE_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["thinking", "node_list"],
    "properties": {
        "thinking": {"type": "string"},
        "node_list": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "pass",
        "score_0_to_1",
        "missing_facts",
        "hallucinations",
        "citation_quality",
        "rationale",
    ],
    "properties": {
        "pass": {"type": "boolean"},
        "score_0_to_1": {"type": "number", "minimum": 0, "maximum": 1},
        "missing_facts": {
            "type": "array",
            "items": {"type": "string"},
        },
        "hallucinations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "citation_quality": {
            "type": "string",
            "enum": sorted(CITATION_LEVELS),
        },
        "rationale": {"type": "string"},
    },
}

TREE_SEARCH_PROMPT = """\
You are given a user question and a PageIndex tree for a single document.
Select up to {top_k} node IDs most likely to contain the evidence needed to answer
the question. Prefer specific nodes over broad parents when possible.

Question:
{question}

PageIndex tree:
{tree}
"""

ANSWER_PROMPT = """\
Answer the question using only the provided document context.

Rules:
- Do not use outside knowledge.
- If the context is insufficient, say so explicitly.
- Cite every material factual claim with the exact source labels from the context
  headers, such as [p3] or [0004 p10-14].
- Keep the answer concise.

Question:
{question}

Document context:
{context}
"""

JUDGE_PROMPT = """\
You are grading a RAG answer against a gold answer.

Question:
{question}

Candidate answer:
{answer}

Reference answer:
{reference_answer}

Required facts:
{required_facts}

Forbidden claims:
{forbidden_claims}

Grading rubric:
- `pass` should be true only if the candidate answer includes all material required
  facts, avoids material contradictions or hallucinations, and includes usable
  citations.
- `score_0_to_1` should reflect overall factual quality and grounding.
- `missing_facts` should list material facts omitted or misstated.
- `hallucinations` should list unsupported or contradictory claims. Return an empty
  list if none.
- `citation_quality` must be one of: missing, weak, adequate, strong.
- `rationale` should be brief and concrete.
"""


@dataclass(frozen=True)
class EvalCase:
    suite: str
    case_id: str
    doc_pdf: Path
    doc_index_json: Path
    question: str
    reference_answer: str
    required_facts: list[str]
    forbidden_claims: list[str]


@dataclass
class DocumentBundle:
    doc_name: str
    pdf_path: Path
    index_path: Path
    page_list: list[tuple[str, int]]
    index_data: dict[str, Any]
    all_nodes: dict[str, dict[str, Any]]
    full_context: str


@dataclass
class EvalResult:
    suite: str
    case_id: str
    question: str
    doc_name: str
    doc_pdf: str
    doc_index_json: str
    model: str
    mode: str
    status: str
    answer: str
    retrieved_node_ids: list[str] = field(default_factory=list)
    retrieval_thinking: str | None = None
    retrieval_usage: dict[str, int] = field(default_factory=dict)
    answer_usage: dict[str, int] = field(default_factory=dict)
    model_usage: dict[str, int] = field(default_factory=dict)
    judge_usage: dict[str, int] = field(default_factory=dict)
    retrieval_latency_sec: float = 0.0
    answer_latency_sec: float = 0.0
    judge_latency_sec: float = 0.0
    total_latency_sec: float = 0.0
    judge: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def zero_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
    }


def add_usage(*usage_items: dict[str, int]) -> dict[str, int]:
    total = zero_usage()
    for usage in usage_items:
        for key in total:
            total[key] += int(usage.get(key, 0))
    return total


def usage_from_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return zero_usage()

    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(output_details, "reasoning_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "cached_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
    }


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def excerpt_text(text: str, limit: int = 220) -> str:
    collapsed = normalize_whitespace(text)
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def resolve_repo_path(raw_path: str) -> Path:
    path = (REPO_ROOT / raw_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return path


def load_dataset(dataset_path: Path) -> list[EvalCase]:
    payload = yaml.safe_load(dataset_path.read_text()) or {}
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"Dataset at {dataset_path} has no cases")

    cases: list[EvalCase] = []
    seen_case_ids: set[str] = set()
    for raw_case in raw_cases:
        required = list(raw_case.get("required_facts") or [])
        forbidden = list(raw_case.get("forbidden_claims") or [])
        case = EvalCase(
            suite=str(raw_case["suite"]),
            case_id=str(raw_case["case_id"]),
            doc_pdf=resolve_repo_path(str(raw_case["doc_pdf"])),
            doc_index_json=resolve_repo_path(str(raw_case["doc_index_json"])),
            question=str(raw_case["question"]),
            reference_answer=str(raw_case["reference_answer"]),
            required_facts=[str(item) for item in required],
            forbidden_claims=[str(item) for item in forbidden],
        )
        if case.case_id in seen_case_ids:
            raise ValueError(f"Duplicate case_id: {case.case_id}")
        seen_case_ids.add(case.case_id)
        if case.suite not in {"smoke", "full"}:
            raise ValueError(f"Unsupported suite '{case.suite}' in case {case.case_id}")
        cases.append(case)

    return cases


def select_cases(cases: list[EvalCase], suite: str) -> list[EvalCase]:
    if suite == "smoke":
        return [case for case in cases if case.suite == "smoke"]
    if suite == "full":
        return [case for case in cases if case.suite in {"smoke", "full"}]
    raise ValueError(f"Unsupported suite: {suite}")


def parse_models(raw_models: str) -> list[str]:
    models = [model.strip() for model in raw_models.split(",") if model.strip()]
    if not models:
        raise ValueError("At least one model must be specified")
    return models


def flatten_nodes(
    structure: list[dict[str, Any]],
    acc: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    if acc is None:
        acc = {}
    for node in structure:
        node_id = node.get("node_id")
        if node_id:
            acc[str(node_id)] = node
        children = node.get("nodes") or []
        if children:
            flatten_nodes(children, acc)
    return acc


def compact_tree(structure: list[dict[str, Any]], depth: int = 0) -> str:
    lines: list[str] = []
    for node in structure:
        node_id = node.get("node_id", "-")
        title = node.get("title", "Untitled")
        start = node.get("start_index", "?")
        end = node.get("end_index", "?")
        summary = node.get("summary") or excerpt_text(str(node.get("text") or ""))
        indent = "  " * depth
        lines.append(f"{indent}[{node_id}] {title} (p{start}-{end})")
        if summary:
            lines.append(f"{indent}  - {excerpt_text(str(summary), limit=180)}")
        children = node.get("nodes") or []
        if children:
            lines.extend(compact_tree(children, depth + 1).splitlines())
    return "\n".join(lines)


def build_full_context(page_list: list[tuple[str, int]]) -> str:
    parts: list[str] = []
    for index, (text, _) in enumerate(page_list, start=1):
        collapsed = normalize_whitespace(text)
        if not collapsed:
            continue
        parts.append(f"[p{index}]\n{collapsed}")
    return "\n\n".join(parts)


def build_pageindex_context(
    node_ids: list[str],
    all_nodes: dict[str, dict[str, Any]],
    *,
    top_k: int = TREE_TOP_K,
) -> str:
    chunks: list[str] = []
    for node_id in unique_preserving_order(node_ids)[:top_k]:
        node = all_nodes.get(node_id)
        if not node:
            continue
        start = node.get("start_index", "?")
        end = node.get("end_index", "?")
        title = node.get("title", "Untitled")
        text = normalize_whitespace(str(node.get("text") or ""))
        if not text:
            continue
        chunks.append(f"[{node_id} p{start}-{end}] {title}\n{text}")
    return "\n\n".join(chunks)


def unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def validate_tree_search(payload: dict[str, Any]) -> dict[str, Any]:
    node_list = payload.get("node_list")
    thinking = payload.get("thinking", "")
    if not isinstance(node_list, list) or not all(isinstance(item, str) for item in node_list):
        raise ValueError("tree_search.node_list must be a list[str]")
    if not isinstance(thinking, str):
        raise ValueError("tree_search.thinking must be a string")
    return {
        "thinking": thinking,
        "node_list": node_list,
    }


def validate_judge_result(payload: dict[str, Any]) -> dict[str, Any]:
    score = payload.get("score_0_to_1")
    citation_quality = payload.get("citation_quality")
    if not isinstance(payload.get("pass"), bool):
        raise ValueError("judge.pass must be a bool")
    if not isinstance(score, (int, float)) or score < 0 or score > 1:
        raise ValueError("judge.score_0_to_1 must be between 0 and 1")
    if citation_quality not in CITATION_LEVELS:
        raise ValueError(f"judge.citation_quality must be one of {sorted(CITATION_LEVELS)}")
    for list_key in ("missing_facts", "hallucinations"):
        items = payload.get(list_key)
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise ValueError(f"judge.{list_key} must be a list[str]")
    rationale = payload.get("rationale")
    if not isinstance(rationale, str):
        raise ValueError("judge.rationale must be a string")
    return {
        "pass": payload["pass"],
        "score_0_to_1": round(float(score), 4),
        "missing_facts": payload["missing_facts"],
        "hallucinations": payload["hallucinations"],
        "citation_quality": citation_quality,
        "rationale": rationale,
    }


class ResponsesRunner:
    def __init__(self, *, api_key: str, client: Any | None = None) -> None:
        self.client = client or OpenAI(api_key=api_key)

    def _ensure_completed(self, response: Any) -> None:
        status = getattr(response, "status", "completed")
        if status in (None, "completed"):
            return
        error = getattr(response, "error", None)
        raise RuntimeError(f"Responses API call did not complete (status={status}, error={error})")

    def call_text(
        self,
        *,
        model: str,
        prompt: str,
        max_output_tokens: int,
    ) -> tuple[str, dict[str, int], float]:
        started = time.perf_counter()
        response = self.client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_output_tokens,
            temperature=0,
            store=False,
            truncation="disabled",
            text={
                "format": {"type": "text"},
                "verbosity": DEFAULT_TEXT_VERBOSITY,
            },
        )
        latency = time.perf_counter() - started
        self._ensure_completed(response)
        return getattr(response, "output_text", "").strip(), usage_from_response(response), latency

    def call_json(
        self,
        *,
        model: str,
        prompt: str,
        format_name: str,
        schema: dict[str, Any],
        validator: Any,
        max_output_tokens: int,
        max_attempts: int = 2,
    ) -> tuple[dict[str, Any], dict[str, int], float]:
        last_error: Exception | None = None
        total_latency = 0.0
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            response = self.client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_output_tokens,
                temperature=0,
                store=False,
                truncation="disabled",
                text={
                    "format": {
                        "type": "json_schema",
                        "name": format_name,
                        "schema": schema,
                        "strict": True,
                    },
                    "verbosity": DEFAULT_TEXT_VERBOSITY,
                },
            )
            latency = time.perf_counter() - started
            total_latency += latency

            try:
                self._ensure_completed(response)
                raw_payload = (getattr(response, "output_text", "") or "").strip()
                parsed = json.loads(raw_payload)
                return validator(parsed), usage_from_response(response), total_latency
            except Exception as exc:  # pragma: no cover - exercised through retry tests
                last_error = exc
                if attempt == max_attempts:
                    break

        assert last_error is not None
        raise RuntimeError(f"Failed to parse structured response after {max_attempts} attempts: {last_error}")


class DocumentStore:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], DocumentBundle] = {}

    def load(self, case: EvalCase) -> DocumentBundle:
        cache_key = (str(case.doc_pdf), str(case.doc_index_json))
        if cache_key in self._cache:
            return self._cache[cache_key]

        page_list = core.get_page_tokens(str(case.doc_pdf), model="gpt-4o")
        index_payload = json.loads(case.doc_index_json.read_text())
        index_data = copy.deepcopy(index_payload)
        core.add_node_text(index_data.get("structure", []), page_list)
        bundle = DocumentBundle(
            doc_name=str(index_data.get("doc_name") or case.doc_pdf.stem),
            pdf_path=case.doc_pdf,
            index_path=case.doc_index_json,
            page_list=page_list,
            index_data=index_data,
            all_nodes=flatten_nodes(index_data.get("structure", [])),
            full_context=build_full_context(page_list),
        )
        self._cache[cache_key] = bundle
        return bundle


def build_tree_search_prompt(case: EvalCase, bundle: DocumentBundle) -> str:
    return TREE_SEARCH_PROMPT.format(
        top_k=TREE_TOP_K,
        question=case.question,
        tree=compact_tree(bundle.index_data.get("structure", [])),
    )


def build_answer_prompt(case: EvalCase, context: str) -> str:
    return ANSWER_PROMPT.format(question=case.question, context=context)


def build_judge_prompt(case: EvalCase, answer: str) -> str:
    required_facts = "\n".join(f"- {fact}" for fact in case.required_facts)
    forbidden_claims = "\n".join(f"- {claim}" for claim in case.forbidden_claims) or "- None"
    return JUDGE_PROMPT.format(
        question=case.question,
        answer=answer,
        reference_answer=case.reference_answer,
        required_facts=required_facts,
        forbidden_claims=forbidden_claims,
    )


def run_tree_search(
    *,
    case: EvalCase,
    bundle: DocumentBundle,
    model: str,
    runner: ResponsesRunner,
) -> tuple[dict[str, Any], dict[str, int], float]:
    result, usage, latency = runner.call_json(
        model=model,
        prompt=build_tree_search_prompt(case, bundle),
        format_name="pageindex_tree_search",
        schema=TREE_SEARCH_SCHEMA,
        validator=validate_tree_search,
        max_output_tokens=TREE_SEARCH_MAX_OUTPUT_TOKENS,
    )
    result["node_list"] = [
        node_id for node_id in unique_preserving_order(result["node_list"])
        if node_id in bundle.all_nodes
    ][:TREE_TOP_K]
    return result, usage, latency


def judge_answer(
    *,
    case: EvalCase,
    answer: str,
    judge_model: str,
    runner: ResponsesRunner,
) -> tuple[dict[str, Any], dict[str, int], float]:
    return runner.call_json(
        model=judge_model,
        prompt=build_judge_prompt(case, answer),
        format_name="pageindex_eval_judge",
        schema=JUDGE_SCHEMA,
        validator=validate_judge_result,
        max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS,
    )


def evaluate_mode(
    *,
    case: EvalCase,
    bundle: DocumentBundle,
    model: str,
    mode: str,
    judge_model: str,
    runner: ResponsesRunner,
) -> EvalResult:
    if mode not in {"full_context", "pageindex"}:
        raise ValueError(f"Unsupported mode: {mode}")

    retrieval_result: dict[str, Any] = {"thinking": "", "node_list": []}
    retrieval_usage = zero_usage()
    retrieval_latency = 0.0

    if mode == "pageindex":
        retrieval_result, retrieval_usage, retrieval_latency = run_tree_search(
            case=case,
            bundle=bundle,
            model=model,
            runner=runner,
        )
        context = build_pageindex_context(retrieval_result["node_list"], bundle.all_nodes)
    else:
        context = bundle.full_context

    answer, answer_usage, answer_latency = runner.call_text(
        model=model,
        prompt=build_answer_prompt(case, context),
        max_output_tokens=ANSWER_MAX_OUTPUT_TOKENS,
    )
    judge_result, judge_usage, judge_latency = judge_answer(
        case=case,
        answer=answer,
        judge_model=judge_model,
        runner=runner,
    )

    model_usage = add_usage(retrieval_usage, answer_usage)
    return EvalResult(
        suite=case.suite,
        case_id=case.case_id,
        question=case.question,
        doc_name=bundle.doc_name,
        doc_pdf=str(case.doc_pdf.relative_to(REPO_ROOT)),
        doc_index_json=str(case.doc_index_json.relative_to(REPO_ROOT)),
        model=model,
        mode=mode,
        status="ok",
        answer=answer,
        retrieved_node_ids=list(retrieval_result["node_list"]),
        retrieval_thinking=str(retrieval_result.get("thinking") or ""),
        retrieval_usage=retrieval_usage,
        answer_usage=answer_usage,
        model_usage=model_usage,
        judge_usage=judge_usage,
        retrieval_latency_sec=round(retrieval_latency, 4),
        answer_latency_sec=round(answer_latency, 4),
        judge_latency_sec=round(judge_latency, 4),
        total_latency_sec=round(retrieval_latency + answer_latency + judge_latency, 4),
        judge=judge_result,
    )


def evaluate_case_mode(
    *,
    case: EvalCase,
    bundle: DocumentBundle,
    model: str,
    mode: str,
    judge_model: str,
    runner: ResponsesRunner,
) -> EvalResult:
    try:
        return evaluate_mode(
            case=case,
            bundle=bundle,
            model=model,
            mode=mode,
            judge_model=judge_model,
            runner=runner,
        )
    except Exception as exc:
        return EvalResult(
            suite=case.suite,
            case_id=case.case_id,
            question=case.question,
            doc_name=bundle.doc_name,
            doc_pdf=str(case.doc_pdf.relative_to(REPO_ROOT)),
            doc_index_json=str(case.doc_index_json.relative_to(REPO_ROOT)),
            model=model,
            mode=mode,
            status="error",
            answer="",
            retrieval_usage=zero_usage(),
            answer_usage=zero_usage(),
            model_usage=zero_usage(),
            judge_usage=zero_usage(),
            error=str(exc),
        )


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_float(value: float) -> str:
    return f"{value:.3f}"


def format_seconds(value: float) -> str:
    return f"{value:.2f}"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator, *body])


def build_summary_matrix(results: list[EvalResult], models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for mode in ("full_context", "pageindex"):
            scoped = [result for result in results if result.model == model and result.mode == mode]
            successful = [result for result in scoped if result.status == "ok" and result.judge]
            if successful:
                pass_rate = mean([1.0 if result.judge["pass"] else 0.0 for result in successful])
                avg_score = mean([float(result.judge["score_0_to_1"]) for result in successful])
                avg_model_tokens = mean([float(result.model_usage["total_tokens"]) for result in successful])
                avg_judge_tokens = mean([float(result.judge_usage["total_tokens"]) for result in successful])
                avg_latency = mean([result.total_latency_sec for result in successful])
            else:
                pass_rate = 0.0
                avg_score = 0.0
                avg_model_tokens = 0.0
                avg_judge_tokens = 0.0
                avg_latency = 0.0

            rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "cases": len(scoped),
                    "successful_cases": len(successful),
                    "pass_rate": round(pass_rate, 4),
                    "avg_score": round(avg_score, 4),
                    "avg_model_tokens": round(avg_model_tokens, 1),
                    "avg_judge_tokens": round(avg_judge_tokens, 1),
                    "avg_total_latency_sec": round(avg_latency, 4),
                    "errors": len(scoped) - len(successful),
                }
            )
    return rows


def build_delta_table(summary_matrix: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    by_key = {(row["model"], row["mode"]): row for row in summary_matrix}
    deltas: list[dict[str, Any]] = []
    for model in models:
        full_row = by_key[(model, "full_context")]
        pageindex_row = by_key[(model, "pageindex")]
        deltas.append(
            {
                "model": model,
                "delta_pass_rate": round(pageindex_row["pass_rate"] - full_row["pass_rate"], 4),
                "delta_avg_score": round(pageindex_row["avg_score"] - full_row["avg_score"], 4),
                "delta_avg_model_tokens": round(
                    pageindex_row["avg_model_tokens"] - full_row["avg_model_tokens"], 1
                ),
                "delta_avg_total_latency_sec": round(
                    pageindex_row["avg_total_latency_sec"] - full_row["avg_total_latency_sec"], 4
                ),
            }
        )
    return deltas


def collect_issue_frequencies(
    results: list[EvalResult],
    issue_key: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for result in results:
        if result.status != "ok" or not result.judge:
            continue
        for issue in result.judge.get(issue_key, []) or []:
            normalized = normalize_whitespace(str(issue))
            if normalized:
                counter[normalized] += 1

    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [{"text": text, "count": count} for text, count in ranked[:limit]]


def choose_strategy_recommendation(
    *,
    full_row: dict[str, Any],
    pageindex_row: dict[str, Any],
) -> tuple[str, str]:
    delta_pass_rate = float(pageindex_row["pass_rate"]) - float(full_row["pass_rate"])
    delta_score = float(pageindex_row["avg_score"]) - float(full_row["avg_score"])
    delta_tokens = float(pageindex_row["avg_model_tokens"]) - float(full_row["avg_model_tokens"])
    delta_latency = float(pageindex_row["avg_total_latency_sec"]) - float(full_row["avg_total_latency_sec"])

    if delta_pass_rate >= 0.15 or delta_score >= 0.1:
        rationale = (
            "Higher judged quality on this suite"
            f" ({format_percent(delta_pass_rate)} pass-rate delta, {format_float(delta_score)} score delta)"
            f" while changing model-token usage by {delta_tokens:.1f}."
        )
        return "PageIndex", rationale

    if delta_pass_rate <= -0.15 or delta_score <= -0.1:
        rationale = (
            "Better judged quality on this suite"
            f" ({format_percent(-delta_pass_rate)} higher pass rate for `full_context`,"
            f" {format_float(-delta_score)} higher score)"
            f" at the cost of {abs(delta_tokens):.1f} additional model tokens."
        )
        return "full_context", rationale

    if delta_tokens <= -1000:
        rationale = (
            "Quality was effectively tied on this suite,"
            f" while PageIndex used {abs(delta_tokens):.1f} fewer model tokens"
            f" and changed latency by {delta_latency:+.2f}s."
        )
        return "PageIndex", rationale

    rationale = (
        "No material quality difference was observed,"
        f" and the token/latency trade-off stayed within a narrow range"
        f" ({delta_tokens:+.1f} model tokens, {delta_latency:+.2f}s)."
    )
    return "Either", rationale


def build_cookbook_analysis(
    *,
    summary_matrix: list[dict[str, Any]],
    results: list[EvalResult],
    models: list[str],
) -> dict[str, Any]:
    by_key = {(row["model"], row["mode"]): row for row in summary_matrix}
    decision_rows: list[dict[str, Any]] = []
    for model in models:
        full_row = by_key[(model, "full_context")]
        pageindex_row = by_key[(model, "pageindex")]
        recommendation, rationale = choose_strategy_recommendation(
            full_row=full_row,
            pageindex_row=pageindex_row,
        )
        decision_rows.append(
            {
                "model": model,
                "full_context": {
                    "pass_rate": full_row["pass_rate"],
                    "avg_score": full_row["avg_score"],
                    "avg_model_tokens": full_row["avg_model_tokens"],
                    "avg_total_latency_sec": full_row["avg_total_latency_sec"],
                },
                "pageindex": {
                    "pass_rate": pageindex_row["pass_rate"],
                    "avg_score": pageindex_row["avg_score"],
                    "avg_model_tokens": pageindex_row["avg_model_tokens"],
                    "avg_total_latency_sec": pageindex_row["avg_total_latency_sec"],
                },
                "delta": {
                    "pass_rate": round(
                        float(pageindex_row["pass_rate"]) - float(full_row["pass_rate"]),
                        4,
                    ),
                    "avg_score": round(
                        float(pageindex_row["avg_score"]) - float(full_row["avg_score"]),
                        4,
                    ),
                    "avg_model_tokens": round(
                        float(pageindex_row["avg_model_tokens"]) - float(full_row["avg_model_tokens"]),
                        1,
                    ),
                    "avg_total_latency_sec": round(
                        float(pageindex_row["avg_total_latency_sec"])
                        - float(full_row["avg_total_latency_sec"]),
                        4,
                    ),
                },
                "recommended_default": recommendation,
                "rationale": rationale,
            }
        )

    missing_facts = collect_issue_frequencies(results, "missing_facts")
    hallucinations = collect_issue_frequencies(results, "hallucinations")
    total_missing = sum(len(result.judge.get("missing_facts", [])) for result in results if result.judge)
    total_hallucinations = sum(len(result.judge.get("hallucinations", [])) for result in results if result.judge)

    return {
        "decision_rows": decision_rows,
        "common_missing_facts": missing_facts,
        "common_hallucinations": hallucinations,
        "total_missing_facts": total_missing,
        "total_hallucinations": total_hallucinations,
        "omission_dominated": total_missing > total_hallucinations,
    }


def render_cookbook_markdown(
    *,
    suite: str,
    dataset_path: Path,
    models: list[str],
    judge_model: str,
    cases: list[EvalCase],
    summary_matrix: list[dict[str, Any]],
    results: list[EvalResult],
    analysis: dict[str, Any],
    generated_at: str,
) -> str:
    decision_rows = [
        [
            row["model"],
            (
                f"{format_percent(row['full_context']['pass_rate'])} / "
                f"{format_float(float(row['full_context']['avg_score']))}"
            ),
            (
                f"{format_percent(row['pageindex']['pass_rate'])} / "
                f"{format_float(float(row['pageindex']['avg_score']))}"
            ),
            f"{row['delta']['avg_model_tokens']:.1f}",
            row["recommended_default"],
        ]
        for row in analysis["decision_rows"]
    ]

    summary_rows = [
        [
            row["model"],
            row["mode"],
            format_percent(row["pass_rate"]),
            format_float(row["avg_score"]),
            f"{row['avg_model_tokens']:.1f}",
            format_seconds(row["avg_total_latency_sec"]),
        ]
        for row in summary_matrix
    ]

    parts = [
        "# Vectorless RAG Strategy Brief",
        "",
        (
            f"This brief summarizes a `{suite}` eval run comparing `full_context` against `pageindex` "
            f"across {len(cases)} PDF QA case(s) and {len(models)} model(s)."
        ),
        "",
        "## Experiment Setup",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Dataset: `{display_path(dataset_path)}`",
        f"- Models under test: `{', '.join(models)}`",
        f"- Judge model: `{judge_model}`",
        "- Modes:",
        "  - `full_context`: inject the full document text into the answer prompt.",
        "  - `pageindex`: retrieve from the PageIndex tree first, then answer from the retrieved nodes.",
        "",
        "## Result Matrix",
        "",
        render_table(
            [
                "Model",
                "Mode",
                "Pass rate",
                "Avg score",
                "Avg model tokens",
                "Avg latency (s)",
            ],
            summary_rows,
        ),
        "",
        "## Default Recommendation By Model",
        "",
        render_table(
            [
                "Model",
                "full_context (pass/score)",
                "PageIndex (pass/score)",
                "Token delta (PageIndex - full)",
                "Recommended default",
            ],
            decision_rows,
        ),
        "",
        *[
            f"- `{row['model']}`: Prefer `{row['recommended_default']}`. {row['rationale']}"
            for row in analysis["decision_rows"]
        ],
        "",
        "## What The Current Results Say",
        "",
    ]

    if suite == "smoke":
        parts.append(
            "- This is the smoke suite, so treat the ranking as directional. Use the `full` suite before "
            "locking a product default or publishing a strong benchmark claim."
        )

    if analysis["omission_dominated"]:
        parts.append(
            "- Current failures are omission-dominated rather than hallucination-dominated. That points to "
            "retrieval coverage and answer completeness as the first tuning targets."
        )

    if analysis["common_missing_facts"]:
        parts.extend(
            [
                "- The most common missing facts in this run were:",
                *[
                    f"  - `{item['text']}` ({item['count']} miss(es))"
                    for item in analysis["common_missing_facts"]
                ],
            ]
        )

    if not analysis["common_hallucinations"]:
        parts.append("- No recurring hallucination pattern stood out in this run.")

    parts.extend(
        [
            "",
            "## When To Use PageIndex As A Retrieval Skill",
            "",
            "- Use PageIndex when the document is too large, too numerous, or too expensive to inject fully into every prompt.",
            "- Use it when the document has meaningful structure (sections, headings, page ranges) that tree search can exploit.",
            "- Use it when traceable node/page citations matter and you want an inspectable retrieval step instead of one giant prompt.",
            "- Keep it as a routed skill in agentic systems: call it only when a document-first retrieval step is likely to beat direct long-context prompting on cost or context pressure.",
            "",
            "## When To Prefer full_context",
            "",
            "- Prefer `full_context` when the document comfortably fits in context and the task depends on recall of cross-section comparison details.",
            "- Prefer it for short or medium single-document tasks where retrieval misses cost more than prompt-token overhead.",
            "- Prefer it when evals show PageIndex dropping material facts that are easy to recover by giving the model the full document.",
            "",
            "## Recommended Next Tuning Steps",
            "",
            "- Expand from the smoke suite to the `full` suite before drawing broad conclusions.",
            "- Tune retrieval and answer prompts to preserve comparative facts such as year-over-year deltas and prior-period baselines.",
            "- Track both quality and token cost in CI, because PageIndex can still be the better operational default when quality is tied and prompt cost drops materially.",
            "",
            "## Reproduce Locally",
            "",
            "```bash",
            "uv sync --dev",
            "export OPENAI_API_KEY=your_openai_key_here",
            "uv run python scripts/eval_matrix.py --suite smoke --output-dir .dist/evals/local-smoke",
            "uv run python scripts/eval_matrix.py --suite full --output-dir .dist/evals/local-full",
            "```",
            "",
            "## OpenAI References",
            "",
            (
                f"- OpenAI recommends integrating graders into CI and iterating on failure modes with an "
                f"evaluation flywheel: {OPENAI_EVAL_FLYWHEEL_URL}"
            ),
            (
                f"- OpenAI also notes that long-context prompting should be evaluated carefully because "
                f"'inject everything' strategies can lose accuracy as prompts grow: "
                f"{OPENAI_OPTIMIZING_ACCURACY_URL}"
            ),
        ]
    )

    return "\n".join(parts).strip() + "\n"


def render_summary_markdown(
    *,
    suite: str,
    dataset_path: Path,
    models: list[str],
    judge_model: str,
    cases: list[EvalCase],
    summary_matrix: list[dict[str, Any]],
    delta_table: list[dict[str, Any]],
    results: list[EvalResult],
    generated_at: str,
) -> str:
    summary_rows = [
        [
            row["model"],
            row["mode"],
            str(row["cases"]),
            format_percent(row["pass_rate"]),
            format_float(row["avg_score"]),
            f"{row['avg_model_tokens']:.1f}",
            f"{row['avg_judge_tokens']:.1f}",
            format_seconds(row["avg_total_latency_sec"]),
            str(row["errors"]),
        ]
        for row in summary_matrix
    ]
    delta_rows = [
        [
            row["model"],
            format_percent(row["delta_pass_rate"]),
            format_float(row["delta_avg_score"]),
            f"{row['delta_avg_model_tokens']:.1f}",
            format_seconds(row["delta_avg_total_latency_sec"]),
        ]
        for row in delta_table
    ]
    error_results = [result for result in results if result.status == "error"]

    parts = [
        "# Eval Matrix Summary",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Suite: `{suite}`",
        f"- Dataset: `{display_path(dataset_path)}`",
        f"- Cases: `{len(cases)}`",
        f"- Models: `{', '.join(models)}`",
        f"- Judge model: `{judge_model}`",
        "",
        "## Summary Matrix",
        "",
        render_table(
            [
                "Model",
                "Mode",
                "Cases",
                "Pass rate",
                "Avg score",
                "Avg model tokens",
                "Avg judge tokens",
                "Avg latency (s)",
                "Errors",
            ],
            summary_rows,
        ),
        "",
        "## Head-to-Head Delta (`PageIndex - full_context`)",
        "",
        render_table(
            [
                "Model",
                "Delta pass rate",
                "Delta avg score",
                "Delta avg model tokens",
                "Delta avg latency (s)",
            ],
            delta_rows,
        ),
    ]

    if error_results:
        parts.extend(
            [
                "",
                "## Errors",
                "",
                *[
                    f"- `{result.model}` / `{result.case_id}` / `{result.mode}`: {result.error}"
                    for result in error_results
                ],
            ]
        )

    return "\n".join(parts).strip() + "\n"


def render_full_report_markdown(
    *,
    summary_markdown: str,
    results: list[EvalResult],
    models: list[str],
    cases: list[EvalCase],
) -> str:
    parts = [summary_markdown.rstrip(), "", "## Per-Case Appendix", ""]
    results_by_case = {(result.case_id, result.model, result.mode): result for result in results}

    for case in cases:
        parts.extend([f"### {case.case_id}", "", f"Question: {case.question}", ""])
        rows: list[list[str]] = []
        notes: list[str] = []
        for model in models:
            for mode in ("full_context", "pageindex"):
                result = results_by_case[(case.case_id, model, mode)]
                if result.status == "ok":
                    rows.append(
                        [
                            model,
                            mode,
                            "pass" if result.judge["pass"] else "fail",
                            format_float(float(result.judge["score_0_to_1"])),
                            result.judge["citation_quality"],
                            str(result.model_usage["total_tokens"]),
                            str(result.judge_usage["total_tokens"]),
                            format_seconds(result.total_latency_sec),
                        ]
                    )
                    note = (
                        f"- `{model}` / `{mode}`: {result.judge['rationale']}"
                        f" Missing facts: {', '.join(result.judge['missing_facts']) or 'none'}."
                        f" Hallucinations: {', '.join(result.judge['hallucinations']) or 'none'}."
                    )
                    if result.retrieved_node_ids:
                        note += f" Retrieved nodes: {', '.join(result.retrieved_node_ids)}."
                    notes.append(note)
                else:
                    rows.append([model, mode, "error", "n/a", "n/a", "0", "0", "0.00"])
                    notes.append(f"- `{model}` / `{mode}`: {result.error}")

        parts.extend(
            [
                render_table(
                    [
                        "Model",
                        "Mode",
                        "Pass",
                        "Score",
                        "Citation quality",
                        "Model tokens",
                        "Judge tokens",
                        "Latency (s)",
                    ],
                    rows,
                ),
                "",
                *notes,
                "",
            ]
        )

    return "\n".join(parts).strip() + "\n"


def build_report(
    *,
    suite: str,
    dataset_path: Path,
    models: list[str],
    judge_model: str,
    cases: list[EvalCase],
    results: list[EvalResult],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    summary_matrix = build_summary_matrix(results, models)
    delta_table = build_delta_table(summary_matrix, models)
    cookbook_analysis = build_cookbook_analysis(
        summary_matrix=summary_matrix,
        results=results,
        models=models,
    )
    summary_markdown = render_summary_markdown(
        suite=suite,
        dataset_path=dataset_path,
        models=models,
        judge_model=judge_model,
        cases=cases,
        summary_matrix=summary_matrix,
        delta_table=delta_table,
        results=results,
        generated_at=generated_at,
    )
    cookbook_markdown = render_cookbook_markdown(
        suite=suite,
        dataset_path=dataset_path,
        models=models,
        judge_model=judge_model,
        cases=cases,
        summary_matrix=summary_matrix,
        results=results,
        analysis=cookbook_analysis,
        generated_at=generated_at,
    )
    report_markdown = render_full_report_markdown(
        summary_markdown=summary_markdown,
        results=results,
        models=models,
        cases=cases,
    )

    return {
        "generated_at": generated_at,
        "suite": suite,
        "dataset": display_path(dataset_path),
        "models": models,
        "judge_model": judge_model,
        "cases": [asdict(case) | {"doc_pdf": str(case.doc_pdf.relative_to(REPO_ROOT)), "doc_index_json": str(case.doc_index_json.relative_to(REPO_ROOT))} for case in cases],
        "summary_matrix": summary_matrix,
        "delta_table": delta_table,
        "analysis": cookbook_analysis,
        "results": [asdict(result) for result in results],
        "summary_markdown": summary_markdown,
        "cookbook_markdown": cookbook_markdown,
        "report_markdown": report_markdown,
    }


def write_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "summary.md").write_text(report["summary_markdown"])
    (output_dir / "cookbook_brief.md").write_text(report["cookbook_markdown"])
    (output_dir / "report.md").write_text(report["report_markdown"])
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))

    for result in report["results"]:
        sample_dir = samples_dir / result["model"] / result["case_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_path = sample_dir / f"{result['mode']}.json"
        sample_path.write_text(json.dumps(result, indent=2))


def execute_run(
    *,
    suite: str,
    dataset_path: Path,
    models: list[str],
    judge_model: str,
    runner: ResponsesRunner,
) -> dict[str, Any]:
    cases = select_cases(load_dataset(dataset_path), suite)
    if not cases:
        raise ValueError(f"No cases selected for suite '{suite}'")

    document_store = DocumentStore()
    results: list[EvalResult] = []

    print(f"Running suite={suite} with {len(cases)} cases across {len(models)} model(s)")
    for model in models:
        for case in cases:
            bundle = document_store.load(case)
            for mode in ("full_context", "pageindex"):
                print(f"  [{model}] {case.case_id} :: {mode}")
                result = evaluate_case_mode(
                    case=case,
                    bundle=bundle,
                    model=model,
                    mode=mode,
                    judge_model=judge_model,
                    runner=runner,
                )
                results.append(result)

    return build_report(
        suite=suite,
        dataset_path=dataset_path,
        models=models,
        judge_model=judge_model,
        cases=cases,
        results=results,
    )


def default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate full-context prompting versus PageIndex retrieval.",
    )
    parser.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--api-key", help="OpenAI API key. Falls back to OPENAI_API_KEY or CHATGPT_API_KEY.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
    if not api_key:
        print("Error: set OPENAI_API_KEY or CHATGPT_API_KEY, or pass --api-key", file=sys.stderr)
        return 1

    dataset_path = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    models = parse_models(args.models)
    runner = ResponsesRunner(api_key=api_key)

    report = execute_run(
        suite=args.suite,
        dataset_path=dataset_path,
        models=models,
        judge_model=args.judge_model,
        runner=runner,
    )
    write_outputs(output_dir, report)

    print(f"Report written to {output_dir}")
    error_count = sum(1 for result in report["results"] if result["status"] == "error")
    if error_count:
        print(f"Completed with {error_count} errored sample(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
