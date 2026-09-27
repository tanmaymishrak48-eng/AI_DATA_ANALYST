"""
Shared pytest fixtures for the AI Data Analyst test suite.
"""

import os
import sys

import pandas as pd
import pytest

# Allow tests to import the project's modules (app.py, ai_engine.py, etc.)
# which live one directory above this tests/ folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SAMPLE_CSV = os.path.join(PROJECT_ROOT, "ai_data_analyst_sample_sales.csv")


@pytest.fixture
def sample_df():
    """Load the bundled sample sales dataset used across tests."""
    return pd.read_csv(SAMPLE_CSV)


# ------------------------------------------------------------------
# Marker for tests that call the live Groq API.
# These are skipped automatically when GROQ_API_KEY is not set,
# so `pytest` still passes cleanly on a machine with no key/network.
# ------------------------------------------------------------------

requires_groq = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping tests that call the live Groq API.",
)
