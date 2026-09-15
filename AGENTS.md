# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12 project providing four dataset converters: `lerobot2mcap/`,
`lerobot2zarr/`, `mcap2lerobot/`, and `hdf52lerobot/`. The unified public facade
is in `xense_dataset_convert/`, with adapters in
`xense_dataset_convert/adapters/`. Shared command-line utilities live in
`tools/`. Put converter-specific documentation in `docs/` and runtime profiles
or conversion rules in `configs/`. Tests belong in `tests/`; avoid placing
generated datasets or large recordings in the repository.

## Build, Test, and Development Commands

Create an editable development install with:

```bash
pip install -e .[dev]
```

The conda recipe (`environment.yml`) also provisions Python and `ffmpeg`.
Run the complete suite with `pytest -q`. Run a focused test while iterating,
for example `pytest -q tests/test_hdf52lerobot.py`. Check lint errors with
`ruff check --select E9,F63,F7,F82 .`. The installed CLIs are
`lerobot2mcap`, `lerobot2zarr`, `mcap2lerobot`, and `hdf52lerobot`; consult the
matching document in `docs/` for representative invocation options.

## Coding Style & Naming Conventions

Use four-space indentation, typed Python interfaces, and concise docstrings for
public APIs and non-obvious conversion rules. Use `snake_case` for functions,
variables, and modules; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for
constants. Keep adapter imports lazy so lightweight facade operations do not
load `lerobot`, Torch, or other heavy dependencies unnecessarily. Format YAML
consistently and update documentation when changing schemas or CLI behavior.

## Testing Guidelines

Tests use `pytest` and should be named `test_*.py` with functions named
`test_*`. Prefer synthetic fixtures and temporary paths so tests do not require
real datasets, hardware, or network access. Add regression coverage for
dispatch, validation, metadata, and format-specific edge cases. The MCAP
validator and encoder benchmark in `tests/` are integration tools and may need
an actual dataset.

## Commit & Pull Request Guidelines

Follow the existing Conventional Commit pattern, using a scope when useful,
such as `feat(hdf52lerobot): ...`, `perf(tools): ...`, `fix: ...`, or
`chore: ...`. Pull requests should explain the format or schema impact, list
tests and lint commands run, call out dependency or configuration changes, and
include before/after examples or sample CLI output when behavior is user-facing.
