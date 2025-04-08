"""
Brown Every Vote Counts 24/25 Analytics Project
=============================

The *src* package contains all reusable, importable code for the project.
Nothing in here should have side‑effects on import.

Sub‑packages
------------
models      • reusable modelling code (ordinal + random‑forest)

Example
-------
>>> from src.models.train_random_forest import main as train_rf
>>> # train_rf(...)  # run CLI programmatically
"""

__all__ = ["models"]
__version__ = "0.1.0"
