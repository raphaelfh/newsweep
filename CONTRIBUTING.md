# Contributing to newsweep

Thanks for your interest in contributing! This guide will help you get started.

## Getting started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/newsweep.git
   cd newsweep
   ```
3. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b my-feature
   ```
2. Make your changes.
3. Run the linter:
   ```bash
   ruff check .
   ruff format --check .
   ```
4. Run the tests:
   ```bash
   pytest
   ```
5. Commit your changes with a clear message.
6. Push to your fork and open a pull request against `main`.

## Code style

- This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- Pre-commit hooks run automatically on each commit to enforce style.
- Follow existing patterns in the codebase.

## Testing

- Tests live in the `tests/` directory and use [pytest](https://docs.pytest.org/).
- Add tests for new functionality -- even simple ones help.
- Make sure all tests pass before opening a PR.

## Pull requests

- Keep PRs small and focused on a single change.
- Write a clear description of what you changed and why.
- Reference related issues if applicable (e.g., "Fixes #12").
- Make sure CI passes (lint + tests).

## Reporting bugs

- Use the [bug report template](https://github.com/raphaelfh/newsweep/issues/new?template=bug_report.md) on GitHub.
- Include your macOS version, chip (M1/M2/M3/M4), Python version, and steps to reproduce.

## Requesting features

- Use the [feature request template](https://github.com/raphaelfh/newsweep/issues/new?template=feature_request.md) on GitHub.
- Describe the problem you're trying to solve, not just the solution.

## Questions?

Open a [discussion](https://github.com/raphaelfh/newsweep/issues) or reach out in an issue.
