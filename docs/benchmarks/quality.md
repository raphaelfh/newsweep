# Quality Benchmark

Measures prediction accuracy by sending realistic editing scenarios to the server and comparing model output against expected completions.

## Usage

```bash
# Run all 100 scenarios
python tests/bench_quality.py --host=localhost --port=8741

# Filter by scenario type
python tests/bench_quality.py --type=refactoring
python tests/bench_quality.py --type=line_completion,block_completion

# Custom scenarios file
python tests/bench_quality.py --scenarios=tests/fixtures/scenarios.json

# Machine-readable output
python tests/bench_quality.py --json --output=results.json
```

## Metrics

| Metric | Description |
|---|---|
| **Exact match rate** | Completion exactly matches expected text (after stripping whitespace) |
| **Prefix match rate** | Model output is a prefix of expected, or expected is a prefix of output |
| **LCP ratio** | Longest common prefix length / expected length — how much the model got right before diverging |
| **Latency** | Per-request time (p50, p95, p99, mean) |

All metrics are reported both globally and per scenario type.

## Scenarios

100 pre-generated scenarios in `tests/fixtures/scenarios.json`, covering:

### Core editing patterns (~48%)

| Type | Count | Example |
|---|---|---|
| `line_completion` | 21 | Completing a dict comprehension filter |
| `block_completion` | 10 | Filling in a function body after the signature |
| `argument_fill` | 10 | Adding arguments inside a function call |
| `import_completion` | 8 | Adding an import statement |
| `string_completion` | 7 | Completing an f-string or log message |

### Refactoring patterns (~20%)

| Type | Count | Example |
|---|---|---|
| `fix_bug` | 6 | Fixing an off-by-one or wrong variable |
| `rename_variable` | 4 | Propagating a rename to subsequent lines |
| `add_type_hint` | 4 | Adding type annotations |
| `extract_method` | 3 | Replacing inline code with a function call |
| `add_parameter` | 3 | Adding a parameter and updating the body |

### Structural patterns (~32%)

| Type | Count | Example |
|---|---|---|
| `new_method_in_class` | 6 | Adding a method consistent with existing ones |
| `add_error_handling` | 5 | Wrapping code in try/except |
| `add_test` | 5 | Writing a test for existing code |
| `docstring` | 4 | Adding a docstring to a function |
| `decorator` | 4 | Adding a decorator |

## Scenario format

Each scenario is a JSON object:

```json
{
  "id": 1,
  "scenario_type": "line_completion",
  "description": "Completing a list comprehension filter",
  "filepath": "src/etl/cleaning.py",
  "file_content": "code before the edit...",
  "cursor_position": 342,
  "prefix": "file_content[:342]",
  "suffix": "file_content[342:]",
  "expected_completion": "if record.get('status') == 'active'",
  "file_content_after": "code after the edit..."
}
```

Invariants (all enforced by validation):
- `prefix + suffix == file_content`
- `cursor_position == len(prefix)`
- `prefix + expected_completion + suffix == file_content_after`
- `file_content_after` is syntactically valid Python

## Generating new scenarios

Use the `autocomplete-testgen` skill in Claude Code to generate additional scenarios:

```
Generate 50 autocomplete test scenarios focused on [pattern]. Save to tests/fixtures/scenarios_custom.json
```

The benchmark automatically loads all `scenarios*.json` files from the fixtures directory.

## Interpreting results

- **Exact match > 30%** is strong for a 7B model — most autocomplete models produce approximate completions.
- **Prefix match > 60%** means the model usually starts correctly even if it diverges later.
- **LCP ratio** is the most nuanced metric — it tells you how much of the completion the developer can accept before needing to edit.
- Compare **latency by type** to find patterns the model struggles with (slower = more tokens generated before early-stop kicks in).
