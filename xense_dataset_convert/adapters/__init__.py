"""Per-format-pair adapters.

Each module here exposes ``ACCEPTED_OPTIONS`` and ``convert(inputs, output,
options)``. None of them imports its converter package at module level -- see
``xense_dataset_convert/registry.py`` for why that matters.
"""
