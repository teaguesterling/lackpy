"""Tests for grammar constants."""

import ast

from lackpy.lang.grammar import ALLOWED_NODES, FORBIDDEN_NODES, FORBIDDEN_NAMES, ALLOWED_BUILTINS


def test_allowed_and_forbidden_are_disjoint():
    overlap = ALLOWED_NODES & FORBIDDEN_NODES
    assert overlap == set(), f"Overlap: {overlap}"


def test_forbidden_names_not_in_builtins():
    overlap = FORBIDDEN_NAMES & ALLOWED_BUILTINS
    assert overlap == set(), f"Overlap: {overlap}"


def test_key_forbidden_nodes_present():
    assert ast.Import in FORBIDDEN_NODES
    assert ast.FunctionDef in FORBIDDEN_NODES
    assert ast.While in FORBIDDEN_NODES
    assert ast.Lambda in FORBIDDEN_NODES
    assert ast.ClassDef in FORBIDDEN_NODES


def test_key_allowed_nodes_present():
    assert ast.Module in ALLOWED_NODES
    assert ast.Assign in ALLOWED_NODES
    assert ast.Call in ALLOWED_NODES
    assert ast.For in ALLOWED_NODES
    assert ast.If in ALLOWED_NODES
    assert ast.Name in ALLOWED_NODES


def test_key_forbidden_names_present():
    assert "__import__" in FORBIDDEN_NAMES
    assert "open" in FORBIDDEN_NAMES
    assert "getattr" in FORBIDDEN_NAMES
    assert "input" in FORBIDDEN_NAMES


def test_key_builtins_present():
    assert "len" in ALLOWED_BUILTINS
    assert "print" in ALLOWED_BUILTINS
    assert "str" in ALLOWED_BUILTINS
    assert "range" in ALLOWED_BUILTINS


def test_sort_by_in_builtins():
    assert "sort_by" in ALLOWED_BUILTINS
