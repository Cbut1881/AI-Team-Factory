"""
Tests for brain.tools -- Dashboard API functions (Section C).

Uses the ``responses`` library to mock HTTP requests to the dashboard.
"""

from __future__ import annotations

import requests
import responses
import pytest

from brain.tools import (
    _DASHBOARD_TIMEOUT,
    _DASHBOARD_URL,
    create_agent,
    create_team,
    delete_agent,
    get_dashboard_system_info,
    get_datasets,
    get_exams,
    list_agents,
    list_models,
    list_teams,
    run_ask,
    run_debate,
    run_parallel,
    run_pipeline,
    train_agent,
    train_distill,
    train_exam,
    train_full,
)

BASE = _DASHBOARD_URL  # e.g. "http://localhost:5555"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_ok(body: dict | list | None = None):
    """Return a dict that the dashboard would typically send back."""
    return body if body is not None else {"ok": True}


# ---------------------------------------------------------------------------
# Dashboard URL and timeout sanity
# ---------------------------------------------------------------------------

class TestDashboardConfig:
    def test_default_url(self):
        assert "localhost" in _DASHBOARD_URL
        assert "5555" in _DASHBOARD_URL

    def test_timeout_positive(self):
        assert _DASHBOARD_TIMEOUT > 0


# ---------------------------------------------------------------------------
# list_agents / list_teams / list_models
# ---------------------------------------------------------------------------

class TestListEndpoints:
    @responses.activate
    def test_list_agents_success(self):
        responses.add(responses.GET, f"{BASE}/api/agents", json=[{"name": "a1"}], status=200)
        result = list_agents()
        assert result["status"] == "ok"
        assert result["result"] == [{"name": "a1"}]

    @responses.activate
    def test_list_teams_success(self):
        responses.add(responses.GET, f"{BASE}/api/teams", json=[{"name": "t1"}], status=200)
        result = list_teams()
        assert result["status"] == "ok"

    @responses.activate
    def test_list_models_success(self):
        responses.add(responses.GET, f"{BASE}/api/models", json=["gpt-4", "llama3"], status=200)
        result = list_models()
        assert result["status"] == "ok"
        assert "gpt-4" in result["result"]

    @responses.activate
    def test_list_agents_http_error(self):
        responses.add(responses.GET, f"{BASE}/api/agents", json={"detail": "fail"}, status=500)
        result = list_agents()
        assert result["status"] == "error"

    @responses.activate
    def test_list_agents_connection_error(self):
        responses.add(responses.GET, f"{BASE}/api/agents", body=requests.ConnectionError("refused"))
        result = list_agents()
        assert result["status"] == "error"
        assert "Cannot connect" in result["error"]


# ---------------------------------------------------------------------------
# create_agent / create_team / delete_agent
# ---------------------------------------------------------------------------

class TestAgentManagement:
    @responses.activate
    def test_create_agent_all_params(self):
        responses.add(responses.POST, f"{BASE}/api/agents", json={"id": 1}, status=200)
        result = create_agent(
            name="coder",
            role="developer",
            model="gpt-4",
            system_prompt="You are a coder.",
            temperature=0.5,
        )
        assert result["status"] == "ok"
        body = responses.calls[0].request.body
        assert b"coder" in body
        assert b"developer" in body

    @responses.activate
    def test_create_agent_defaults(self):
        responses.add(responses.POST, f"{BASE}/api/agents", json={"id": 2}, status=200)
        result = create_agent(name="bot", role="helper", model="llama3")
        assert result["status"] == "ok"

    @responses.activate
    def test_create_agent_connection_error(self):
        responses.add(responses.POST, f"{BASE}/api/agents", body=requests.ConnectionError())
        result = create_agent(name="x", role="y", model="z")
        assert result["status"] == "error"
        assert "Cannot connect" in result["error"]

    @responses.activate
    def test_create_team_success(self):
        responses.add(responses.POST, f"{BASE}/api/teams", json={"id": 1}, status=200)
        result = create_team(name="alpha", agents=["a1", "a2"], workflow="parallel")
        assert result["status"] == "ok"

    @responses.activate
    def test_create_team_defaults(self):
        responses.add(responses.POST, f"{BASE}/api/teams", json={"id": 2}, status=200)
        result = create_team(name="beta", agents=["a1"])
        assert result["status"] == "ok"

    @responses.activate
    def test_delete_agent_success(self):
        responses.add(responses.DELETE, f"{BASE}/api/agents/old_bot", json={"deleted": True}, status=200)
        result = delete_agent("old_bot")
        assert result["status"] == "ok"

    @responses.activate
    def test_delete_agent_not_found(self):
        responses.add(responses.DELETE, f"{BASE}/api/agents/ghost", json={"detail": "not found"}, status=404)
        result = delete_agent("ghost")
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

class TestRunModes:
    @responses.activate
    def test_run_ask(self):
        responses.add(responses.POST, f"{BASE}/api/run/ask", json={"answer": "42"}, status=200)
        result = run_ask(agent_name="bot", prompt="What is the meaning?")
        assert result["status"] == "ok"
        assert result["result"]["answer"] == "42"

    @responses.activate
    def test_run_pipeline(self):
        responses.add(responses.POST, f"{BASE}/api/run/pipeline", json={"output": "done"}, status=200)
        result = run_pipeline(team_name="team1", prompt="Summarize")
        assert result["status"] == "ok"

    @responses.activate
    def test_run_parallel(self):
        responses.add(responses.POST, f"{BASE}/api/run/parallel", json={"results": ["a", "b"]}, status=200)
        result = run_parallel(team_name="team1", prompt="Brainstorm")
        assert result["status"] == "ok"

    @responses.activate
    def test_run_debate(self):
        responses.add(responses.POST, f"{BASE}/api/run/debate", json={"winner": "a1"}, status=200)
        result = run_debate(team_name="team1", prompt="Topic", rounds=5)
        assert result["status"] == "ok"

    @responses.activate
    def test_run_debate_default_rounds(self):
        responses.add(responses.POST, f"{BASE}/api/run/debate", json={"winner": "a1"}, status=200)
        result = run_debate(team_name="team1", prompt="Topic")
        assert result["status"] == "ok"
        import json
        body = json.loads(responses.calls[0].request.body)
        assert body["rounds"] == 3

    @responses.activate
    def test_run_ask_connection_error(self):
        responses.add(responses.POST, f"{BASE}/api/run/ask", body=requests.ConnectionError())
        result = run_ask(agent_name="bot", prompt="hi")
        assert result["status"] == "error"
        assert "Cannot connect" in result["error"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestTraining:
    @responses.activate
    def test_train_distill(self):
        responses.add(responses.POST, f"{BASE}/api/train/distill", json={"job": 1}, status=200)
        result = train_distill(
            teacher_model="gpt-4",
            student_model="llama3",
            dataset="ds1",
            epochs=5,
        )
        assert result["status"] == "ok"

    @responses.activate
    def test_train_distill_defaults(self):
        responses.add(responses.POST, f"{BASE}/api/train/distill", json={"job": 2}, status=200)
        result = train_distill(teacher_model="t", student_model="s", dataset="d")
        assert result["status"] == "ok"
        import json
        body = json.loads(responses.calls[0].request.body)
        assert body["epochs"] == 3

    @responses.activate
    def test_train_agent(self):
        responses.add(responses.POST, f"{BASE}/api/train/agent", json={"job": 1}, status=200)
        result = train_agent(agent_name="bot", dataset="ds1", epochs=10)
        assert result["status"] == "ok"

    @responses.activate
    def test_train_exam(self):
        responses.add(responses.POST, f"{BASE}/api/train/exam", json={"score": 85}, status=200)
        result = train_exam(agent_name="bot", exam="math_eval")
        assert result["status"] == "ok"

    @responses.activate
    def test_train_full(self):
        responses.add(responses.POST, f"{BASE}/api/train/full", json={"score": 90}, status=200)
        result = train_full(agent_name="bot", dataset="ds1", exam="eval1", epochs=2)
        assert result["status"] == "ok"

    @responses.activate
    def test_train_full_defaults(self):
        responses.add(responses.POST, f"{BASE}/api/train/full", json={"score": 90}, status=200)
        result = train_full(agent_name="bot", dataset="ds1", exam="eval1")
        assert result["status"] == "ok"

    @responses.activate
    def test_train_distill_connection_error(self):
        responses.add(responses.POST, f"{BASE}/api/train/distill", body=requests.ConnectionError())
        result = train_distill(teacher_model="t", student_model="s", dataset="d")
        assert result["status"] == "error"
        assert "Cannot connect" in result["error"]


# ---------------------------------------------------------------------------
# Dashboard info endpoints
# ---------------------------------------------------------------------------

class TestDashboardInfo:
    @responses.activate
    def test_get_dashboard_system_info(self):
        responses.add(responses.GET, f"{BASE}/api/system_info", json={"cpu": "ok"}, status=200)
        result = get_dashboard_system_info()
        assert result["status"] == "ok"
        assert result["result"]["cpu"] == "ok"

    @responses.activate
    def test_get_datasets(self):
        responses.add(responses.GET, f"{BASE}/api/datasets", json=["ds1", "ds2"], status=200)
        result = get_datasets()
        assert result["status"] == "ok"
        assert "ds1" in result["result"]

    @responses.activate
    def test_get_exams(self):
        responses.add(responses.GET, f"{BASE}/api/exams", json=["exam1"], status=200)
        result = get_exams()
        assert result["status"] == "ok"
        assert "exam1" in result["result"]

    @responses.activate
    def test_get_datasets_connection_error(self):
        responses.add(responses.GET, f"{BASE}/api/datasets", body=requests.ConnectionError())
        result = get_datasets()
        assert result["status"] == "error"
        assert "Cannot connect" in result["error"]

    @responses.activate
    def test_get_exams_server_error(self):
        responses.add(responses.GET, f"{BASE}/api/exams", json={"error": "fail"}, status=500)
        result = get_exams()
        assert result["status"] == "error"
