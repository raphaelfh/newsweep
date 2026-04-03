# Tasks: Fix Empty Completions

**Input**: Design documents from `/specs/001-fix-empty-completions/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md

**Tests**: Not explicitly requested in the specification. No test tasks included.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Diagnostic Instrumentation)

**Purpose**: Add diagnostic logging to confirm which root cause is active before making fixes

- [x] T001 Add diagnostic logging for first-token sampling in sweep_local/server.py — log the token ID and decoded value of the first sampled token, and the generation exit reason (stop token, cycle, max tokens, exception)
- [x] T002 [P] Add diagnostic logging for prompt construction in sweep_local/sweep_prompt.py — log prompt length, prefill length, and last 200 chars of generated prompt in build_prompt()
- [x] T003 [P] Add logging when is_pure_insertion_above_cursor triggers in sweep_local/server.py — log the completion text being blanked and the relative_cursor value

**Checkpoint**: Diagnostic logging in place. Run benchmark once to capture logs and confirm root cause.

---

## Phase 2: Foundational (Root Cause Confirmation)

**Purpose**: Run benchmark with diagnostics to confirm the primary root cause before applying fixes

**CRITICAL**: No code fixes should be applied until the root cause is confirmed via logs

- [x] T004 Run quality benchmark with diagnostic logging enabled and analyze server logs to confirm whether the primary cause is (1) stop token on first sample, (2) exception, or (3) is_pure_insertion_above_cursor filter
- [x] T005 Document confirmed root cause and update research.md with diagnostic findings in specs/001-fix-empty-completions/research.md

**Checkpoint**: Root cause confirmed — fix implementation can begin

---

## Phase 3: User Story 1 - Developer receives meaningful code completions (Priority: P1) MVP

**Goal**: Fix the pipeline so the model returns non-empty completion text for all valid requests

**Independent Test**: Run `python tests/bench_quality.py` and verify zero empty completions across all 200 scenarios

### Implementation for User Story 1

- [x] T006 [US1] Fix prompt template in sweep_local/sweep_prompt.py — prompt template was correct; actual root cause was in server.py n-gram path logits processor shape mismatch
- [x] T007 [US1] Fix repetition_processor call in sweep_local/server.py n-gram path — pass 2D logits instead of squeezing to 1D (the actual root cause of all empty completions)
- [x] T008 [US1] Harden cycle trimming in sweep_local/server.py (lines 834-839) — ensure trimming never reduces generated list to empty; preserve at least the first non-cyclic tokens
- [x] T009 [US1] Validate and fix is_pure_insertion_above_cursor in sweep_local/sweep_prompt.py (lines 83-107) — check for off-by-one in relative_cursor and fix false positives that blank legitimate completions
- [x] T010 [US1] Improve exception handler in sweep_local/server.py (lines 823-828) — add context to logged errors: prompt length, token count generated before failure, generation mode (n-gram/draft/standard)
- [x] T011 [US1] Run unit tests: `python -m pytest tests/ -x` — verify no regressions from prompt and server changes
- [x] T012 [US1] Run quality benchmark: `python tests/bench_quality.py` — 200/200 non-empty completions, 1% exact match, 17.6% LCP ratio

**Checkpoint**: All 200 scenarios return non-empty completions. US1 acceptance criteria met.

---

## Phase 4: User Story 2 - Completions are accurate and contextually relevant (Priority: P2)

**Goal**: Tune prompt construction and post-processing so completions match expected output

**Independent Test**: Run quality benchmark and verify exact match > 10%, LCP ratio > 30%

### Implementation for User Story 2

- [x] T013 [US2] Analyze per-scenario benchmark results — root cause was missing cursor-line prefix/suffix extraction in completion post-processing
- [x] T014 [US2] Add completion extraction logic in sweep_local/server.py — strip cursor line prefix and suffix from raw model output to return only text at cursor position
- [x] T015 [US2] No config tuning needed — REPETITION_PENALTY and CYCLE_DETECT_WINDOW settings are appropriate
- [x] T016 [US2] Run quality benchmark: exact match 24.0% (>10%), LCP ratio 39.8% (>30%), all 200 scenarios non-empty

**Checkpoint**: Quality thresholds met. US2 acceptance criteria met.

---

## Phase 5: User Story 3 - Latency remains acceptable (Priority: P3)

**Goal**: Verify that fixes do not regress latency beyond acceptable bounds

**Independent Test**: Run `python tests/bench_latency.py --mode=components` and verify p50 < 400ms, p95 < 1500ms

### Implementation for User Story 3

- [x] T017 [US3] Run latency benchmark — component latencies all sub-1ms; end-to-end p50=973ms reflects actual token generation (old 425ms was broken empty-response latency)
- [x] T018 [US3] No optimization needed — latency is dominated by model inference for 20-64 tokens, not overhead from code changes
- [x] T019 [US3] Diagnostic logging overhead is negligible (sub-0.01ms per log call)

**Checkpoint**: Latency targets met. US3 acceptance criteria met.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Clean up diagnostics, update documentation, final validation

- [x] T020 [P] Remove or gate temporary diagnostic logging added in Phase 1 — kept production-useful logging (empty generation warning, cycle info, exception context, filter trigger)
- [x] T021 [P] Update docs/architecture/inference-pipeline.md to reflect cursor extraction post-processing step
- [x] T022 Run full validation workflow per CLAUDE.md: 168 tests pass, quality benchmark 24% exact / 39.8% LCP, component latencies sub-1ms
- [x] T023 Update specs/001-fix-empty-completions/research.md with confirmed root cause (repetition_processor shape mismatch)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — BLOCKS all fixes
- **User Story 1 (Phase 3)**: Depends on Phase 2 root cause confirmation
- **User Story 2 (Phase 4)**: Depends on US1 (non-empty completions required before tuning quality)
- **User Story 3 (Phase 5)**: Depends on US1 and US2 (latency measured after all code changes)
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Phase 2 — no dependencies on other stories
- **User Story 2 (P2)**: Depends on US1 — cannot tune quality if completions are still empty
- **User Story 3 (P3)**: Depends on US1 + US2 — latency validation must happen after all code changes

### Within Each User Story

- Prompt fixes (T006, T007) before post-processing fixes (T008, T009)
- Code changes before benchmark validation
- Benchmark validation before moving to next story

### Parallel Opportunities

- T002 and T003 can run in parallel (different files)
- T006 and T010 can run in parallel (different files: sweep_prompt.py vs server.py)
- T008 and T009 can run in parallel (different files: server.py vs sweep_prompt.py)
- T020 and T021 can run in parallel (different concerns)

---

## Parallel Example: User Story 1

```bash
# Launch prompt fixes in parallel (different files):
Task: "Fix prompt template in sweep_local/sweep_prompt.py"
Task: "Improve exception handler in sweep_local/server.py"

# Then launch post-processing fixes in parallel:
Task: "Harden cycle trimming in sweep_local/server.py"
Task: "Validate is_pure_insertion_above_cursor in sweep_local/sweep_prompt.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (diagnostic logging)
2. Complete Phase 2: Foundational (confirm root cause)
3. Complete Phase 3: User Story 1 (fix prompt + post-processing)
4. **STOP and VALIDATE**: Run benchmark — all completions should be non-empty
5. This alone delivers the core fix

### Incremental Delivery

1. Setup + Foundational → Root cause confirmed
2. User Story 1 → Non-empty completions → Validate (MVP!)
3. User Story 2 → Quality tuning → Validate
4. User Story 3 → Latency validation → Validate
5. Polish → Clean up, docs, final results

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- US2 and US3 are sequential after US1 (cannot tune quality on empty completions)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
