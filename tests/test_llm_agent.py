"""
Unit tests for app/llm_agent.py with mocked Gemini API interactions (no network calls).
Verifies tool dispatching, grounding validation, numerical checks, and rule-based fallback.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.llm_agent import LLMAgent, verify_grounding, extract_numbers_from_text


def test_fallback_triggers_when_no_api_key_set(temp_db):
    """Verifies that an agent initialized with no API key uses rule-based fallback."""
    agent = LLMAgent(api_key="", db_path=temp_db)
    res = agent.ask("What is our total net revenue?")

    assert res["status"] == "success"
    assert res["source"] == "rule_based"
    assert res["fallback_reason"] == "no_api_key"
    assert "total net revenue" in res["answer"].lower()
    assert res["exact_tool"] == "analytics.get_total_net_revenue()"


def test_fallback_triggers_on_llm_exception(temp_db):
    """Verifies that network / LLM exceptions trigger graceful rule-based fallback."""
    agent = LLMAgent(api_key="fake-test-key", db_path=temp_db)

    with patch.object(agent, "call_gemini_api", side_effect=RuntimeError("API Network Timeout")):
        res = agent.ask("Which product generated the highest net revenue?")

        assert res["status"] == "success"
        assert res["source"] == "rule_based"
        assert res["fallback_reason"] == "timeout"
        assert "monitor" in res["answer"].lower()


def test_grounding_validation_accepts_valid_payload():
    """Verifies that valid narratives whose numbers are present in tool output pass grounding."""
    narrative = "The total net revenue was 1445200.00 across 50 completed orders with 4500.00 in discounts."
    payload = {
        "total_net_revenue": 1445200.0,
        "completed_orders_count": 50,
        "total_discounts": 4500.0
    }
    assert verify_grounding(narrative, payload) is True


def test_grounding_validation_rejects_hallucinated_number():
    """Verifies that narratives containing made-up figures fail the grounding check."""
    # 9999999.00 was never in the payload
    narrative = "The total net revenue was 9999999.00 across 50 completed orders."
    payload = {
        "total_net_revenue": 1445200.0,
        "completed_orders_count": 50
    }
    assert verify_grounding(narrative, payload) is False


def test_llm_agent_grounding_triggers_templated_fallback(temp_db):
    """
    Verifies that when Gemini returns a tool call followed by a hallucinated narrative,
    the agent replaces the hallucinated narrative with a verified templated narrative.
    """
    agent = LLMAgent(api_key="fake-test-key", db_path=temp_db)

    # Mock tool call response followed by hallucinated text response
    tool_call_resp = {
        "candidates": [{
            "content": {
                "parts": [{
                    "functionCall": {
                        "name": "get_total_net_revenue",
                        "args": {}
                    }
                }]
            }
        }]
    }

    hallucinated_resp = {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": "Total revenue is 88888888.88 from 99999 fake orders."
                }]
            }
        }]
    }

    with patch.object(agent, "call_gemini_api", side_effect=[tool_call_resp, hallucinated_resp]):
        res = agent.ask("What is our total net revenue?")
        assert res["status"] == "success"
        # The answer must NOT contain the hallucinated 88888888.88
        assert "88888888.88" not in res["answer"]
        # It must contain the real number from the deterministic template
        assert "14,45,200.00" in res["answer"]


def test_ambiguity_returns_needs_clarification(temp_db):
    """Verifies that ambiguous month comparisons immediately return needs_clarification."""
    agent = LLMAgent(api_key="fake-test-key", db_path=temp_db)
    res = agent.ask("how did revenue change between two months")

    assert res["status"] == "needs_clarification"
    assert "specify" in res["answer"].lower()


def test_dispatch_tool_run_sql(temp_db):
    """Verifies that run_sql tool correctly executes safe SQL."""
    agent = LLMAgent(api_key="fake-test-key", db_path=temp_db)
    rows, tool_ref = agent.dispatch_tool("run_sql", {"query": "SELECT COUNT(*) as cnt FROM transactions"})
    assert len(rows) == 1
    assert rows[0]["cnt"] == 51
    assert "SELECT COUNT(*)" in tool_ref


def test_bm25_glossary_retrieval_aov_and_returns():
    """Phase 5 Test: Verifies vectorless BM25 retrieval for 'AOV' and 'returns'."""
    from app.llm_agent import BM25GlossaryRetriever
    retriever = BM25GlossaryRetriever()

    # Search for AOV
    aov_results = retriever.retrieve("What is our AOV?", top_k=3)
    aov_titles = [r["title"] for r in aov_results]
    assert any("Average Order Value" in t for t in aov_titles)

    # Search for returns
    return_results = retriever.retrieve("Show me returns metrics", top_k=3)
    return_titles = [r["title"] for r in return_results]
    assert any("Returned Orders" in t for t in return_titles)


def test_rapidfuzz_entity_normalization_wireles_mouse():
    """Phase 5 Test: Verifies that 'wireles mouse' resolves to 'Wireless Mouse'."""
    from app.llm_agent import normalize_query_entities
    normalized_q, replacements = normalize_query_entities("how many wireles mouse were sold in the East?")
    assert "Wireless Mouse" in normalized_q
    assert "wireles mouse" in replacements
    assert replacements["wireles mouse"] == "Wireless Mouse"


def test_api_response_includes_glossary_entries(temp_db):
    """Phase 5 Test: Verifies that ask() returns glossary_entries."""
    agent = LLMAgent(api_key="", db_path=temp_db)
    res = agent.ask("What is our total net revenue?")
    assert "glossary_entries" in res
    assert isinstance(res["glossary_entries"], list)
    assert len(res["glossary_entries"]) > 0


def test_api_key_never_appears_in_logged_messages_when_api_raises(caplog):
    """Verifies that API key is completely redacted and never logged when API raises."""
    secret_key = "MOCK-SECRET-TEST-KEY-1234567890"
    agent = LLMAgent(api_key=secret_key)

    with patch("httpx.Client.post", side_effect=Exception(f"Connection error to API with header x-goog-api-key: {secret_key}")):
        with pytest.raises(RuntimeError) as exc_info:
            agent.call_gemini_api([{"role": "user", "parts": [{"text": "hi"}]}], "sys prompt")

        # Key must not appear in exception message
        assert secret_key not in str(exc_info.value)
        # Key must not appear anywhere in captured logs
        assert secret_key not in caplog.text


def test_rejected_sql_fed_back_and_corrected_second_attempt_succeeds(temp_db):
    """
    Verifies bounded loop: rejected SQL error is fed back to model as functionResponse,
    and a corrected second attempt succeeds.
    """
    agent = LLMAgent(api_key="mock-key", db_path=temp_db)

    # 1. Model attempts invalid SQL (DROP)
    step1_resp = {
        "candidates": [{
            "content": {
                "parts": [{
                    "functionCall": {
                        "name": "run_sql",
                        "args": {"query": "DROP TABLE transactions; --"}
                    }
                }]
            }
        }]
    }

    # 2. Model corrects itself with valid SELECT
    step2_resp = {
        "candidates": [{
            "content": {
                "parts": [{
                    "functionCall": {
                        "name": "run_sql",
                        "args": {"query": "SELECT product, SUM(net_revenue) as net_revenue FROM transactions WHERE status = 'Completed' GROUP BY product"}
                    }
                }]
            }
        }]
    }

    # 3. Model synthesizes grounded response
    step3_resp = {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": "Products generated revenue across completed transactions."
                }]
            }
        }]
    }

    with patch.object(agent, "call_gemini_api", side_effect=[step1_resp, step2_resp, step3_resp]):
        res = agent.ask("Show revenue by product")

        assert res["status"] == "success"
        assert res["source"] == "llm_agent"
        assert "SELECT product" in res["exact_tool"]
        assert len(res["supporting_rows"]) > 0


def test_persistent_failure_returns_cannot_answer(temp_db):
    """
    Verifies that persistent tool errors across all bounded loop steps
    return status 'cannot_answer' with a clear message and no guessing.
    """
    agent = LLMAgent(api_key="mock-key", db_path=temp_db)

    # Model stubbornly tries invalid SQL 3 times
    failing_resp = {
        "candidates": [{
            "content": {
                "parts": [{
                    "functionCall": {
                        "name": "run_sql",
                        "args": {"query": "SELECT * FROM non_existent_table"}
                    }
                }]
            }
        }]
    }

    with patch.object(agent, "call_gemini_api", side_effect=[failing_resp, failing_resp, failing_resp]):
        res = agent.ask("Show me secret table")

        assert res["status"] == "cannot_answer"
        assert "could not be completed" in res["answer"].lower()
        assert res["supporting_rows"] == []


def test_assumptions_reflect_run_sql_query_without_status_filter(temp_db):
    """
    Verifies that assumptions for a run_sql query without status='Completed'
    explicitly say 'Custom SQL: verify filters'.
    """
    from app.llm_agent import build_assumptions_from_execution

    # 1. Query without status='Completed'
    assumptions_no_filter = build_assumptions_from_execution(
        "run_sql",
        "SELECT product, quantity FROM transactions",
        {"query": "SELECT product, quantity FROM transactions"}
    )
    assert assumptions_no_filter == ["Custom SQL: verify filters"]

    # 2. Query with status='Completed'
    assumptions_with_filter = build_assumptions_from_execution(
        "run_sql",
        "SELECT product FROM transactions WHERE status = 'Completed'",
        {"query": "SELECT product FROM transactions WHERE status = 'Completed'"}
    )
    assert any("Filtered for Completed orders" in a for a in assumptions_with_filter)


def test_grounding_failure_for_run_sql_builds_table_summary_never_generic(temp_db):
    """
    Verifies that when grounding fails on run_sql results, the templated narrative
    is built directly from the executed SQL rows (short table summary),
    and never from the rule-based intent or generic placeholder.
    """
    agent = LLMAgent(api_key="mock-key", db_path=temp_db)

    tool_call_resp = {
        "candidates": [{
            "content": {
                "parts": [{
                    "functionCall": {
                        "name": "run_sql",
                        "args": {"query": "SELECT category, SUM(net_revenue) as net_revenue FROM transactions WHERE status = 'Completed' GROUP BY category"}
                    }
                }]
            }
        }]
    }

    # Model hallucinates a figure (8888888.00) that isn't in the SQL result
    hallucinated_resp = {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": "The category revenue was 8888888.00."
                }]
            }
        }]
    }

    with patch.object(agent, "call_gemini_api", side_effect=[tool_call_resp, hallucinated_resp]):
        res = agent.ask("Which category made the most money?")

        assert res["status"] == "success"
        # Must not contain hallucinated number
        assert "8888888.00" not in res["answer"]
        # Must be built from the tool that actually ran (run_sql table summary)
        assert "Query returned" in res["answer"]
        # Never the generic placeholder
        assert "Query processed successfully" not in res["answer"]


def test_candidate_content_with_thought_signature_preserved_in_turn(temp_db):
    """
    Verifies that thoughtSignature and functionCall id in candidate content
    are preserved and passed back to the model without being dropped.
    """
    agent = LLMAgent(api_key="mock-key", db_path=temp_db)

    # Candidate with thoughtSignature and id
    mock_model_content = {
        "role": "model",
        "parts": [{
            "functionCall": {
                "name": "get_total_net_revenue",
                "args": {},
                "id": "call_12345"
            },
            "thoughtSignature": "valid_signature_token"
        }]
    }

    mock_resp1 = {"candidates": [{"content": mock_model_content}]}
    mock_resp2 = {
        "candidates": [{
            "content": {
                "parts": [{"text": "Total revenue is 1445200.00."}]
            }
        }]
    }

    recorded_contents = []

    def mock_call(contents, prompt, tools=None):
        recorded_contents.append(list(contents))
        if len(recorded_contents) == 1:
            return mock_resp1
        return mock_resp2

    with patch.object(agent, "call_gemini_api", side_effect=mock_call):
        res = agent.ask("What is our total net revenue?")
        assert res["status"] == "success"

        # Check recorded turn in second call
        second_call_contents = recorded_contents[1]
        # Must have model turn with exact thoughtSignature
        model_turn = second_call_contents[1]
        assert model_turn["parts"][0]["thoughtSignature"] == "valid_signature_token"
        # Must have functionResponse with matching call_id
        func_turn = second_call_contents[2]
        assert func_turn["parts"][0]["functionResponse"]["id"] == "call_12345"




