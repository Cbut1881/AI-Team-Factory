"""
Tests for brain.tools -- Tool dispatch and TOOL_DEFINITIONS / TOOL_FUNCTIONS.

Validates execute_tool routing, error handling, and schema consistency.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from brain.tools import (
    TOOL_DEFINITIONS,
    TOOL_FUNCTIONS,
    execute_tool,
)


# ---------------------------------------------------------------------------
# execute_tool dispatch
# ---------------------------------------------------------------------------

class TestExecuteTool:
    def test_dispatch_click(self):
        mock_click = MagicMock(return_value={"status": "ok", "result": "clicked"})
        with patch.dict(TOOL_FUNCTIONS, {"click": mock_click}):
            result = execute_tool("click", {"x": 10, "y": 20, "button": "left"})
        mock_click.assert_called_once_with(x=10, y=20, button="left")
        assert result["status"] == "ok"

    def test_dispatch_screenshot(self):
        mock_ss = MagicMock(return_value={"status": "ok"})
        with patch.dict(TOOL_FUNCTIONS, {"screenshot": mock_ss}):
            result = execute_tool("screenshot", {})
        mock_ss.assert_called_once_with()
        assert result["status"] == "ok"

    def test_dispatch_list_agents(self):
        mock_la = MagicMock(return_value={"status": "ok", "result": []})
        with patch.dict(TOOL_FUNCTIONS, {"list_agents": mock_la}):
            result = execute_tool("list_agents", {})
        mock_la.assert_called_once_with()
        assert result["status"] == "ok"

    def test_unknown_tool(self):
        result = execute_tool("nonexistent_tool", {})
        assert result["status"] == "error"
        assert "Unknown tool" in result["error"]

    def test_wrong_arguments_type_error(self):
        mock_fn = MagicMock(side_effect=TypeError("bad arg"))
        with patch.dict(TOOL_FUNCTIONS, {"click": mock_fn}):
            result = execute_tool("click", {"x": 10, "y": 20, "bad_param": True})
        assert result["status"] == "error"
        assert "Invalid arguments" in result["error"]

    def test_generic_exception(self):
        mock_fn = MagicMock(side_effect=RuntimeError("something broke"))
        with patch.dict(TOOL_FUNCTIONS, {"click": mock_fn}):
            result = execute_tool("click", {"x": 1, "y": 2})
        assert result["status"] == "error"
        assert "failed" in result["error"]

    def test_dispatch_hotkey_with_keys_list(self):
        """execute_tool passes keys=["ctrl","s"] via **kwargs.

        hotkey() uses *keys (VAR_POSITIONAL), so hotkey(keys=[...]) raises
        TypeError.  execute_tool catches this and returns an error envelope.
        This documents the known dispatch mismatch for hotkey.
        """
        result = execute_tool("hotkey", {"keys": ["ctrl", "s"]})
        assert result["status"] == "error"
        assert "Invalid arguments" in result["error"]


# ---------------------------------------------------------------------------
# TOOL_FUNCTIONS registry
# ---------------------------------------------------------------------------

class TestToolFunctions:
    def test_all_values_are_callable(self):
        for name, func in TOOL_FUNCTIONS.items():
            assert callable(func), f"TOOL_FUNCTIONS['{name}'] is not callable"

    def test_known_tools_present(self):
        expected = [
            "click", "double_click", "right_click", "type_text", "hotkey",
            "mouse_move", "scroll", "open_application", "get_mouse_position",
            "screenshot", "webcam_capture",
            "list_agents", "list_teams", "list_models",
            "create_agent", "create_team", "delete_agent",
            "run_ask", "run_pipeline", "run_parallel", "run_debate",
            "train_distill", "train_agent", "train_exam", "train_full",
            "get_dashboard_system_info", "get_datasets", "get_exams",
            "run_command", "open_url", "get_clipboard", "set_clipboard",
            "file_read", "file_write", "get_system_info",
        ]
        for name in expected:
            assert name in TOOL_FUNCTIONS, f"Missing tool function: {name}"


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS schema validity
# ---------------------------------------------------------------------------

class TestToolDefinitions:
    def test_definitions_count_matches_functions(self):
        assert len(TOOL_DEFINITIONS) == len(TOOL_FUNCTIONS)

    def test_names_match_between_definitions_and_functions(self):
        def_names = {d["name"] for d in TOOL_DEFINITIONS}
        func_names = set(TOOL_FUNCTIONS.keys())
        assert def_names == func_names, (
            f"Mismatch -- in definitions only: {def_names - func_names}, "
            f"in functions only: {func_names - def_names}"
        )

    def test_each_definition_has_required_fields(self):
        for defn in TOOL_DEFINITIONS:
            assert "name" in defn, f"Definition missing 'name': {defn}"
            assert "description" in defn, f"Definition missing 'description': {defn.get('name')}"
            assert "input_schema" in defn, f"Definition missing 'input_schema': {defn.get('name')}"

    def test_input_schema_structure(self):
        for defn in TOOL_DEFINITIONS:
            schema = defn["input_schema"]
            assert schema.get("type") == "object", (
                f"Tool '{defn['name']}' input_schema type should be 'object'"
            )
            assert "properties" in schema, (
                f"Tool '{defn['name']}' input_schema missing 'properties'"
            )
            assert "required" in schema, (
                f"Tool '{defn['name']}' input_schema missing 'required'"
            )
            assert isinstance(schema["required"], list), (
                f"Tool '{defn['name']}' input_schema 'required' should be a list"
            )

    def test_description_is_nonempty_string(self):
        for defn in TOOL_DEFINITIONS:
            assert isinstance(defn["description"], str), (
                f"Tool '{defn['name']}' description is not a string"
            )
            assert len(defn["description"]) > 0, (
                f"Tool '{defn['name']}' has empty description"
            )

    def test_name_is_nonempty_string(self):
        for defn in TOOL_DEFINITIONS:
            assert isinstance(defn["name"], str) and len(defn["name"]) > 0


# ---------------------------------------------------------------------------
# Hotkey dispatch bug check
# ---------------------------------------------------------------------------

class TestHotkeyDispatchBug:
    def test_hotkey_signature_accepts_varargs(self):
        """hotkey() uses *keys, so calling hotkey("ctrl","s") works.

        But execute_tool does func(**arguments), so hotkey(keys=["ctrl","s"])
        will be called.  This is a known mismatch -- verify the current
        behavior so it is documented.
        """
        import inspect
        from brain.tools import hotkey

        sig = inspect.signature(hotkey)
        params = list(sig.parameters.values())
        # hotkey has a single VAR_POSITIONAL parameter (*keys)
        assert params[0].kind == inspect.Parameter.VAR_POSITIONAL, (
            "hotkey should accept *keys (VAR_POSITIONAL)"
        )
        # Since execute_tool does func(**{"keys": [...]}) and hotkey takes
        # *keys, Python will raise TypeError because 'keys' is not a valid
        # keyword argument for a function that only accepts *args.
        # This documents the dispatch bug.
        try:
            from brain.tools import hotkey as hk
            hk(keys=["ctrl", "s"])
            bug_exists = False
        except TypeError:
            bug_exists = True
        assert bug_exists, (
            "Expected TypeError when calling hotkey(keys=[...]) -- "
            "the dispatch uses **kwargs but hotkey takes *args"
        )
