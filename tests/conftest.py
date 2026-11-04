"""Shared test setup: a non-interactive backend and an ``ax`` fixture."""

import matplotlib as mpl

mpl.use("Agg")  # before pyplot is imported anywhere

import matplotlib.pyplot as plt
import pytest


@pytest.fixture
def ax():
    """A fresh axes, closed after the test."""
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)
