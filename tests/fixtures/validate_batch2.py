"""Validate all scenarios against the skill's invariant rules."""
import ast
import json
import sys

with open("/Users/raphael/PycharmProjects/newsweep/tests/fixtures/scenarios_batch2.json") as f:
    scenarios = json.load(f)

errors = []
file_contents_seen = set()

for s in scenarios:
    sid = s["id"]
    prefix = s["prefix"]
    suffix = s["suffix"]
    fc = s["file_content"]
    fca = s["file_content_after"]
    cp = s["cursor_position"]
    ec = s["expected_completion"]

    # Rule 1: prefix + suffix == file_content
    if prefix + suffix != fc:
        errors.append(f"Scenario {sid}: prefix + suffix != file_content")
        if len(prefix) + len(suffix) != len(fc):
            errors.append(f"  lengths: prefix={len(prefix)} + suffix={len(suffix)} = {len(prefix)+len(suffix)}, fc={len(fc)}")

    # Rule 2: cursor_position == len(prefix)
    if cp != len(prefix):
        errors.append(f"Scenario {sid}: cursor_position ({cp}) != len(prefix) ({len(prefix)})")

    # Rule 3: prefix + expected_completion + suffix == file_content_after
    if prefix + ec + suffix != fca:
        errors.append(f"Scenario {sid}: prefix + expected_completion + suffix != file_content_after")
        reconstructed = prefix + ec + suffix
        if len(reconstructed) != len(fca):
            errors.append(f"  lengths: reconstructed={len(reconstructed)}, fca={len(fca)}")
        else:
            for i, (a, b) in enumerate(zip(reconstructed, fca)):
                if a != b:
                    errors.append(f"  first diff at pos {i}: got {a!r}, expected {b!r}")
                    errors.append(f"  context: ...{reconstructed[max(0,i-20):i+20]!r}...")
                    break

    # Rule 4: file_content_after parses with ast.parse()
    try:
        ast.parse(fca)
    except SyntaxError as e:
        errors.append(f"Scenario {sid}: file_content_after fails ast.parse(): {e}")

    # Rule 5: 10 <= len(expected_completion) <= 300
    eclen = len(ec)
    if eclen < 10 or eclen > 300:
        errors.append(f"Scenario {sid}: expected_completion length {eclen} outside [10, 300]")

    # Rule 6: No two scenarios share the same file_content
    if fc in file_contents_seen:
        errors.append(f"Scenario {sid}: duplicate file_content")
    file_contents_seen.add(fc)

print(f"Validated {len(scenarios)} scenarios")

# Summary by type
from collections import Counter
type_counts = Counter(s["scenario_type"] for s in scenarios)
print("\nScenario type distribution:")
for st, count in sorted(type_counts.items()):
    print(f"  {st}: {count}")

# ID range check
ids = [s["id"] for s in scenarios]
print(f"\nID range: {min(ids)}-{max(ids)}")

if errors:
    print(f"\nFAILED with {len(errors)} errors:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("\nALL CHECKS PASSED")
