"""
Root conftest.py — prevents task_decomposition_and_planning/tests/test_lab.py
from being collected during the main project's pytest run.

The task_decomposition_and_planning/ directory is a self-contained sub-project
with its own .git and its own requirements (langchain_mistralai). Its tests
import from 'planning_lab' which is only resolvable when pytest is run from
*inside* task_decomposition_and_planning/, not from the repo root.

Collecting it from the root causes:
    ModuleNotFoundError: No module named 'planning_lab'
which aborts the entire test session. This conftest ignores that subtree.
"""

collect_ignore_glob = ["task_decomposition_and_planning/*"]
