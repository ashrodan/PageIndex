import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from scripts import eval_matrix


class FakeResponse:
    def __init__(
        self,
        output_text: str,
        *,
        input_tokens: int = 10,
        output_tokens: int = 5,
        reasoning_tokens: int = 0,
        cached_tokens: int = 0,
        status: str = "completed",
        error=None,
    ):
        self.output_text = output_text
        self.status = status
        self.error = error
        self.usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
            output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        )


class FakeResponsesAPI:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("No fake responses left")
        next_item = self._responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


class FakeClient:
    def __init__(self, responses):
        self.responses = FakeResponsesAPI(responses)


def _first_case(case_id: str) -> eval_matrix.EvalCase:
    cases = eval_matrix.load_dataset(eval_matrix.DEFAULT_DATASET_PATH)
    return next(case for case in cases if case.case_id == case_id)


class TestDatasetLoading:
    def test_suite_filtering_smoke_and_full(self):
        cases = eval_matrix.load_dataset(eval_matrix.DEFAULT_DATASET_PATH)
        smoke = eval_matrix.select_cases(cases, "smoke")
        full = eval_matrix.select_cases(cases, "full")

        assert len(smoke) == 3
        assert len(full) == 12
        assert {case.case_id for case in smoke}.issubset({case.case_id for case in full})


class TestDocumentEnrichment:
    def test_document_store_adds_node_text_in_memory_only(self):
        case = _first_case("disney-q1-revenue")
        store = eval_matrix.DocumentStore()
        bundle = store.load(case)

        assert bundle.full_context.startswith("[p1]")
        assert bundle.all_nodes["0002"]["text"]

        source_payload = json.loads(case.doc_index_json.read_text())
        source_nodes = eval_matrix.flatten_nodes(source_payload["structure"])
        assert "text" not in source_nodes["0002"]


class TestPromptBuilders:
    def test_answer_prompt_requires_exact_context_citations(self):
        case = _first_case("disney-q1-revenue")
        prompt = eval_matrix.build_answer_prompt(case, "[p1]\nRevenue was $24.7 billion.")

        assert "Do not use outside knowledge." in prompt
        assert "such as [p3] or [0004 p10-14]" in prompt
        assert case.question in prompt

    def test_tree_search_prompt_includes_top_k_and_question(self):
        case = _first_case("disney-q1-revenue")
        bundle = eval_matrix.DocumentStore().load(case)
        prompt = eval_matrix.build_tree_search_prompt(case, bundle)

        assert str(eval_matrix.TREE_TOP_K) in prompt
        assert case.question in prompt
        assert "PageIndex tree" in prompt


class TestStructuredResponses:
    def test_call_json_retries_on_malformed_output(self):
        runner = eval_matrix.ResponsesRunner(
            api_key="test",
            client=FakeClient(
                [
                    FakeResponse("not-json"),
                    FakeResponse(json.dumps({"thinking": "ok", "node_list": ["0001"]})),
                ]
            ),
        )

        parsed, usage, latency = runner.call_json(
            model="gpt-4.1-mini",
            prompt="tree",
            format_name="tree",
            schema=eval_matrix.TREE_SEARCH_SCHEMA,
            validator=eval_matrix.validate_tree_search,
            max_output_tokens=100,
        )

        assert parsed["node_list"] == ["0001"]
        assert usage["total_tokens"] == 15
        assert latency >= 0
        assert runner.client.responses.calls[0]["text"]["verbosity"] == eval_matrix.DEFAULT_TEXT_VERBOSITY
        assert runner.client.responses.calls[1]["text"]["verbosity"] == eval_matrix.DEFAULT_TEXT_VERBOSITY


class TestAggregation:
    def test_summary_and_delta_tables(self):
        results = [
            eval_matrix.EvalResult(
                suite="smoke",
                case_id="c1",
                question="q",
                doc_name="doc",
                doc_pdf="tests/pdfs/q1-fy25-earnings.pdf",
                doc_index_json="tests/results/q1-fy25-earnings_structure.json",
                model="gpt-4o-mini",
                mode="full_context",
                status="ok",
                answer="a",
                model_usage={**eval_matrix.zero_usage(), "total_tokens": 100},
                judge_usage={**eval_matrix.zero_usage(), "total_tokens": 20},
                total_latency_sec=2.0,
                judge={"pass": False, "score_0_to_1": 0.4, "citation_quality": "adequate", "missing_facts": [], "hallucinations": [], "rationale": "x"},
            ),
            eval_matrix.EvalResult(
                suite="smoke",
                case_id="c1",
                question="q",
                doc_name="doc",
                doc_pdf="tests/pdfs/q1-fy25-earnings.pdf",
                doc_index_json="tests/results/q1-fy25-earnings_structure.json",
                model="gpt-4o-mini",
                mode="pageindex",
                status="ok",
                answer="a",
                model_usage={**eval_matrix.zero_usage(), "total_tokens": 70},
                judge_usage={**eval_matrix.zero_usage(), "total_tokens": 20},
                total_latency_sec=1.5,
                judge={"pass": True, "score_0_to_1": 0.9, "citation_quality": "strong", "missing_facts": [], "hallucinations": [], "rationale": "y"},
            ),
        ]

        summary = eval_matrix.build_summary_matrix(results, ["gpt-4o-mini"])
        delta = eval_matrix.build_delta_table(summary, ["gpt-4o-mini"])

        assert summary[0]["mode"] == "full_context"
        assert summary[1]["mode"] == "pageindex"
        assert delta[0]["delta_avg_score"] == pytest.approx(0.5)
        assert delta[0]["delta_avg_model_tokens"] == pytest.approx(-30.0)

    def test_cookbook_analysis_prefers_pageindex_on_quality_tie_with_token_savings(self):
        results = [
            eval_matrix.EvalResult(
                suite="smoke",
                case_id="c1",
                question="q",
                doc_name="doc",
                doc_pdf="tests/pdfs/q1-fy25-earnings.pdf",
                doc_index_json="tests/results/q1-fy25-earnings_structure.json",
                model="gpt-4o-mini",
                mode="full_context",
                status="ok",
                answer="a",
                model_usage={**eval_matrix.zero_usage(), "total_tokens": 2500},
                judge_usage={**eval_matrix.zero_usage(), "total_tokens": 20},
                total_latency_sec=2.0,
                judge={
                    "pass": True,
                    "score_0_to_1": 0.9,
                    "citation_quality": "strong",
                    "missing_facts": [],
                    "hallucinations": [],
                    "rationale": "x",
                },
            ),
            eval_matrix.EvalResult(
                suite="smoke",
                case_id="c1",
                question="q",
                doc_name="doc",
                doc_pdf="tests/pdfs/q1-fy25-earnings.pdf",
                doc_index_json="tests/results/q1-fy25-earnings_structure.json",
                model="gpt-4o-mini",
                mode="pageindex",
                status="ok",
                answer="a",
                model_usage={**eval_matrix.zero_usage(), "total_tokens": 900},
                judge_usage={**eval_matrix.zero_usage(), "total_tokens": 20},
                total_latency_sec=2.4,
                judge={
                    "pass": True,
                    "score_0_to_1": 0.9,
                    "citation_quality": "strong",
                    "missing_facts": ["Revenue was up 5% from the prior-year quarter."],
                    "hallucinations": [],
                    "rationale": "y",
                },
            ),
        ]

        summary = eval_matrix.build_summary_matrix(results, ["gpt-4o-mini"])
        analysis = eval_matrix.build_cookbook_analysis(
            summary_matrix=summary,
            results=results,
            models=["gpt-4o-mini"],
        )

        assert analysis["decision_rows"][0]["recommended_default"] == "PageIndex"
        assert analysis["common_missing_facts"][0]["text"] == "Revenue was up 5% from the prior-year quarter."
        assert analysis["omission_dominated"] is True


class TestIntegration:
    def test_execute_run_and_write_outputs_with_mocked_client(self, tmp_path):
        dataset_path = tmp_path / "dataset.yaml"
        dataset_path.write_text(
            "\n".join(
                [
                    "version: 1",
                    "cases:",
                    "  - suite: smoke",
                    "    case_id: disney-q1-revenue",
                    "    doc_pdf: tests/pdfs/q1-fy25-earnings.pdf",
                    "    doc_index_json: tests/results/q1-fy25-earnings_structure.json",
                    "    question: What total revenue did Disney report for the first quarter of fiscal 2025?",
                    "    reference_answer: Disney reported $24.7 billion of revenue in the first quarter of fiscal 2025.",
                    "    required_facts:",
                    "      - Disney reported $24.7 billion in revenue.",
                    "    forbidden_claims:",
                    "      - Revenue declined year over year.",
                ]
            )
        )

        fake_client = FakeClient(
            [
                FakeResponse("Disney reported $24.7 billion in revenue. [p1]"),
                FakeResponse(json.dumps({
                    "pass": True,
                    "score_0_to_1": 0.95,
                    "missing_facts": [],
                    "hallucinations": [],
                    "citation_quality": "strong",
                    "rationale": "Matches the gold answer.",
                })),
                FakeResponse(json.dumps({"thinking": "earnings table", "node_list": ["0002", "0008"]})),
                FakeResponse("Disney reported $24.7 billion in revenue. [0002 p1-1]"),
                FakeResponse(json.dumps({
                    "pass": True,
                    "score_0_to_1": 0.96,
                    "missing_facts": [],
                    "hallucinations": [],
                    "citation_quality": "strong",
                    "rationale": "Correct and grounded.",
                })),
            ]
        )
        runner = eval_matrix.ResponsesRunner(api_key="test", client=fake_client)

        report = eval_matrix.execute_run(
            suite="smoke",
            dataset_path=dataset_path,
            models=["gpt-4o-mini"],
            judge_model="gpt-4.1",
            runner=runner,
        )
        eval_matrix.write_outputs(tmp_path, report)

        report_json = json.loads((tmp_path / "report.json").read_text())
        report_md = (tmp_path / "report.md").read_text()
        summary_md = (tmp_path / "summary.md").read_text()
        cookbook_md = (tmp_path / "cookbook_brief.md").read_text()
        sample_full = tmp_path / "samples" / "gpt-4o-mini" / "disney-q1-revenue" / "full_context.json"
        sample_pageindex = tmp_path / "samples" / "gpt-4o-mini" / "disney-q1-revenue" / "pageindex.json"

        assert report_json["summary_matrix"][0]["mode"] == "full_context"
        assert report_json["summary_matrix"][1]["mode"] == "pageindex"
        assert "analysis" in report_json
        assert "Eval Matrix Summary" in report_md
        assert "Head-to-Head Delta" in summary_md
        assert "Vectorless RAG Strategy Brief" in cookbook_md
        assert "When To Use PageIndex As A Retrieval Skill" in cookbook_md
        assert sample_full.exists()
        assert sample_pageindex.exists()
        assert all(
            call["text"]["verbosity"] == eval_matrix.DEFAULT_TEXT_VERBOSITY
            for call in fake_client.responses.calls
        )
