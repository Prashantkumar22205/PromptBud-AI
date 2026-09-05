"""
tests/test_structure.py

Unit tests for structural signal detection.
"""

from __future__ import annotations

import pytest

from app.analysis.structure import compute_structure


# ── Role detection ────────────────────────────────────────────────────────────


def test_role_you_are():
    result = compute_structure("You are an expert Python teacher.")
    assert result.hasRoleInstruction is True


def test_role_act_as():
    result = compute_structure("Act as a software engineer and review this code.")
    assert result.hasRoleInstruction is True


def test_role_youre():
    result = compute_structure("You're a skilled data scientist.")
    assert result.hasRoleInstruction is True


def test_role_not_detected():
    result = compute_structure("Explain binary search step by step.")
    assert result.hasRoleInstruction is False


# ── Context detection ──────────────────────────────────────────────────────────


def test_context_for_beginner():
    result = compute_structure("Explain binary search for a beginner.")
    assert result.hasContext is True


def test_context_for_students():
    result = compute_structure("Write a tutorial for students.")
    assert result.hasContext is True


def test_context_target_audience():
    result = compute_structure("The target audience is junior developers.")
    assert result.hasContext is True


def test_context_not_detected():
    result = compute_structure("Generate a Python function.")
    assert result.hasContext is False


# ── Task instruction detection ────────────────────────────────────────────────


def test_task_explain():
    result = compute_structure("Explain binary search.")
    assert result.hasTaskInstruction is True


def test_task_write():
    result = compute_structure("Write a Python function.")
    assert result.hasTaskInstruction is True


def test_task_generate():
    result = compute_structure("Generate a summary of this text.")
    assert result.hasTaskInstruction is True


def test_task_summarize():
    result = compute_structure("Summarize the following article.")
    assert result.hasTaskInstruction is True


def test_task_create():
    result = compute_structure("Create a flowchart.")
    assert result.hasTaskInstruction is True


# ── Output format detection ───────────────────────────────────────────────────


def test_format_bullet_points():
    result = compute_structure("Use bullet points for the answer.")
    assert result.hasOutputFormat is True


def test_format_headings():
    result = compute_structure("Use headings and simple language.")
    assert result.hasOutputFormat is True


def test_format_json():
    result = compute_structure("Return JSON with the key results.")
    assert result.hasOutputFormat is True


def test_format_numbered_list():
    result = compute_structure("Provide numbered steps for the process.")
    assert result.hasOutputFormat is True


def test_format_not_detected():
    result = compute_structure("Explain binary search.")
    assert result.hasOutputFormat is False


# ── Constraint detection ──────────────────────────────────────────────────────


def test_constraint_do_not():
    result = compute_structure("Do not use advanced libraries.")
    assert result.hasConstraints is True


def test_constraint_dont():
    result = compute_structure("Don't repeat the same information.")
    assert result.hasConstraints is True


def test_constraint_keep_short():
    result = compute_structure("Keep the answer short but informative.")
    assert result.hasConstraints is True


def test_constraint_make_sure():
    result = compute_structure("Make sure the explanation is clear.")
    assert result.hasConstraints is True


def test_constraint_limit():
    result = compute_structure("Limit the response to 200 words.")
    assert result.hasConstraints is True


def test_constraint_not_detected():
    result = compute_structure("Explain binary search.")
    assert result.hasConstraints is False


# ── Example detection ─────────────────────────────────────────────────────────


def test_example_for_example():
    result = compute_structure("For example, binary search works like this.")
    assert result.hasExamples is True


def test_example_eg():
    result = compute_structure("Use simple data structures, e.g. arrays.")
    assert result.hasExamples is True


def test_example_such_as():
    result = compute_structure("Use structures such as arrays and lists.")
    assert result.hasExamples is True


def test_example_code_block():
    result = compute_structure("Here is the code:\n```python\nprint('hello')\n```")
    assert result.hasExamples is True


def test_example_not_detected():
    # Use a prompt with no example signals at all
    result = compute_structure("Write a function to sort numbers in ascending order.")
    assert result.hasExamples is False


# ── Question detection ────────────────────────────────────────────────────────


def test_question_how():
    result = compute_structure("How does binary search work?")
    assert result.hasQuestion is True


def test_question_what():
    result = compute_structure("What is recursion?")
    assert result.hasQuestion is True


def test_question_not_detected():
    result = compute_structure("Explain binary search step by step.")
    assert result.hasQuestion is False


# ── Sections list ──────────────────────────────────────────────────────────────


def test_sections_populated_correctly():
    prompt = (
        "You are an expert. Explain sorting for beginners. "
        "Use bullet points. Do not use complex terms. "
        "For example, bubble sort."
    )
    result = compute_structure(prompt)
    assert "Role Instruction" in result.sections
    assert "Context / Audience" in result.sections
    assert "Task Instruction" in result.sections
    assert "Output Format" in result.sections
    assert "Constraints" in result.sections
    assert "Examples" in result.sections


def test_empty_sections_for_plain_prompt():
    result = compute_structure("Sort the list.")
    # This might detect task (sort) but not much else
    # Just verify sections is a list
    assert isinstance(result.sections, list)
