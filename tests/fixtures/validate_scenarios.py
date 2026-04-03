#!/usr/bin/env python3
"""Validate all scenarios in scenarios_batch1.json against the skill's invariants."""
import json
import ast
import sys
from pathlib import Path

def validate(path: str) -> bool:
    with open(path) as f:
        scenarios = json.load(f)

    errors = []
    seen_contents = set()
    all_passed = True

    for s in scenarios:
        sid = s["id"]
        prefix = s["prefix"]
        suffix = s["suffix"]
        fc = s["file_content"]
        fca = s["file_content_after"]
        cp = s["cursor_position"]
        ec = s["expected_completion"]

        # Invariant 1: prefix + suffix == file_content
        if prefix + suffix != fc:
            errors.append(f"ID {sid}: FAIL invariant 1 - prefix + suffix != file_content")
            all_passed = False

        # Invariant 2: cursor_position == len(prefix)
        if cp != len(prefix):
            errors.append(f"ID {sid}: FAIL invariant 2 - cursor_position ({cp}) != len(prefix) ({len(prefix)})")
            all_passed = False

        # Invariant 3: prefix + expected_completion + suffix == file_content_after
        if prefix + ec + suffix != fca:
            errors.append(f"ID {sid}: FAIL invariant 3 - prefix + expected_completion + suffix != file_content_after")
            # Debug info
            reconstructed = prefix + ec + suffix
            if len(reconstructed) != len(fca):
                errors.append(f"  Length mismatch: reconstructed={len(reconstructed)}, expected={len(fca)}")
            else:
                for i, (a, b) in enumerate(zip(reconstructed, fca)):
                    if a != b:
                        errors.append(f"  First diff at char {i}: got {repr(a)}, expected {repr(b)}")
                        errors.append(f"  Context: ...{repr(reconstructed[max(0,i-20):i+20])}...")
                        break
            all_passed = False

        # Invariant 4: file_content_after parses with ast.parse()
        try:
            ast.parse(fca)
        except SyntaxError as e:
            errors.append(f"ID {sid}: FAIL invariant 4 - SyntaxError: {e}")
            all_passed = False

        # Invariant 5: 10 <= len(expected_completion) <= 300
        ec_len = len(ec)
        if not (10 <= ec_len <= 300):
            errors.append(f"ID {sid}: FAIL invariant 5 - len(expected_completion) = {ec_len}")
            all_passed = False

        # Invariant 6: no duplicate file_content
        if fc in seen_contents:
            errors.append(f"ID {sid}: FAIL invariant 6 - duplicate file_content")
            all_passed = False
        seen_contents.add(fc)

    # Print results
    if errors:
        for e in errors:
            print(e)

    total = len(scenarios)
    failed = len([e for e in errors if "FAIL" in e])
    passed = total - len(set(int(e.split(":")[0].replace("ID ", "").strip()) for e in errors if "FAIL" in e))
    print(f"\n{'='*50}")
    print(f"Total scenarios: {total}")
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    if all_passed:
        print("ALL INVARIANTS PASSED")

    return all_passed

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/raphael/PycharmProjects/newsweep/tests/fixtures/scenarios_batch1.json"
    success = validate(path)
    sys.exit(0 if success else 1)
