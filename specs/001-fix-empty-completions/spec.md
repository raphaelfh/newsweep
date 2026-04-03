# Feature Specification: Fix Empty Completions

**Feature Branch**: `001-fix-empty-completions`  
**Created**: 2026-04-03  
**Status**: Draft  
**Input**: User description: "The model returns empty completions for every scenario — 100% prefix match (empty string is a prefix of everything) but 0% exact match and 0% LCP ratio. The server is responding (latency ~425ms p50), but the actual completion content is empty across all 200 scenarios. Research to plan improvements to fix the code."

## Clarifications

### Session 2026-04-03

- Q: Should the spec scope include fixing prompt construction (root cause) or stay narrow to post-processing? → A: Broaden scope to include prompt construction fixes (address root cause + post-processing guards).
- Q: Should generation errors be observable instead of silently swallowed? → A: Log errors with context, still return empty completion to client (non-breaking).
- Q: Should the `is_pure_insertion_above_cursor` filter be investigated as part of this fix? → A: Investigate and fix if overly aggressive (in scope).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Developer receives meaningful code completions (Priority: P1)

A developer using a JetBrains IDE types code and pauses, triggering an autocomplete request to the local server. The server returns a non-empty, contextually relevant code completion that helps the developer continue writing code faster.

**Why this priority**: This is the core value proposition of the product. If completions are empty, the entire system provides zero value. Fixing this unblocks all other quality improvements.

**Independent Test**: Run the quality benchmark (`python tests/bench_quality.py`) against the live server and verify that completions are non-empty strings that match or partially match expected outputs.

**Acceptance Scenarios**:

1. **Given** the server is running and a completion request is sent with valid prefix/suffix context, **When** the model generates tokens, **Then** the response contains a non-empty completion text.
2. **Given** a benchmark suite of 200 realistic editing scenarios, **When** all scenarios are executed, **Then** zero completions are empty strings (100% non-empty).
3. **Given** the model generates repetitive output, **When** cycle detection triggers, **Then** the response still contains meaningful (non-empty) completion text derived from the tokens generated before the cycle began.

---

### User Story 2 - Completions are accurate and contextually relevant (Priority: P2)

A developer writing Python code receives completions that are syntactically valid and contextually appropriate for the code surrounding the cursor position. The completion should predict what the developer would actually type next.

**Why this priority**: Non-empty completions that are wrong or irrelevant are only marginally better than empty ones. Quality matters for developer trust and adoption.

**Independent Test**: Run the quality benchmark and verify that exact match rate and LCP ratio are above baseline thresholds (exact match > 10%, LCP ratio > 30%).

**Acceptance Scenarios**:

1. **Given** a line completion scenario with clear surrounding context, **When** the server generates a completion, **Then** the completion is syntactically valid for the target language.
2. **Given** the full benchmark suite, **When** results are aggregated, **Then** the mean LCP ratio exceeds 30%.
3. **Given** the full benchmark suite, **When** results are aggregated, **Then** the exact match rate exceeds 10%.

---

### User Story 3 - Latency remains acceptable after fixes (Priority: P3)

After fixing the empty completions issue, the server continues to respond within acceptable latency bounds so that the developer experience remains smooth and non-disruptive.

**Why this priority**: Latency is already acceptable (~425ms p50). Fixes should not regress this. However, a slightly slower but working system is better than a fast but broken one.

**Independent Test**: Run `python tests/bench_latency.py --mode=components` and verify p50 latency stays under 400ms.

**Acceptance Scenarios**:

1. **Given** the fix is applied, **When** the latency benchmark runs, **Then** p50 latency remains under 400ms.
2. **Given** the fix is applied, **When** the latency benchmark runs, **Then** p95 latency remains under 1500ms.

---

### Edge Cases

- What happens when the model genuinely produces only repetitive tokens (degenerate output)? The system should return the best non-empty prefix before the repetition started, not an empty string.
- What happens when the model produces very short output (1-3 tokens) that happens to match the cycle detection window? The system should not trim output below a minimum useful length.
- What happens when the prompt is malformed or missing context? The system should log the error with context and return empty gracefully, rather than silently swallowing the exception.
- What happens when `is_pure_insertion_above_cursor` incorrectly classifies a valid completion? The filter should be validated to avoid false positives that blank legitimate output.
- What happens when the model's first predicted token is a stop/special token due to prompt formatting? The system should not silently return empty — the prompt construction should be corrected to avoid this condition.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST construct prompts so the model produces meaningful code tokens rather than immediately predicting a stop/special token. The prompt format MUST correctly condition the model for code completion.
- **FR-002**: System MUST return non-empty completion text when the model has generated at least one meaningful token. Specifically:
  - When cycle detection triggers, at least the first non-cyclic tokens MUST be preserved.
  - Post-generation trimming MUST NOT reduce the decoded result to an empty string or whitespace-only.
- **FR-003**: System MUST still detect and handle degenerate repetitive output to avoid returning excessively long repeated sequences.
- **FR-004**: System MUST log generation errors with sufficient context (prompt length, error type, scenario metadata) instead of silently returning empty completions. Error logging MUST NOT change the client-facing API response format.
- **FR-005**: The `is_pure_insertion_above_cursor` filter MUST be validated to ensure it does not incorrectly blank legitimate completions. If found overly aggressive, it MUST be relaxed or corrected.
- **FR-006**: System MUST maintain the existing Tabby-compatible API contract — response format, endpoint paths, and field names remain unchanged.
- **FR-007**: System MUST pass the quality benchmark with exact match rate > 10% and LCP ratio > 30% across the 200-scenario test suite.
- **FR-008**: System MUST maintain p50 latency under 400ms after the fix is applied.

### Key Entities

- **Completion Request**: Contains prefix, suffix, filepath, file content, and cursor position describing the editing context.
- **Generated Token Stream**: The sequence of tokens produced by the model during inference, subject to cycle detection and trimming.
- **Completion Response**: The final decoded text returned to the IDE client after post-processing.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Zero scenarios in the 200-scenario benchmark return empty completions (down from 200/200 currently).
- **SC-002**: Exact match rate exceeds 10% on the quality benchmark (up from 0%).
- **SC-003**: Mean LCP ratio exceeds 30% (up from 0%).
- **SC-004**: Median completion latency (p50) remains under 400ms.
- **SC-005**: 95th percentile latency remains under 1500ms.
- **SC-006**: All existing unit tests continue to pass.

## Assumptions

- The underlying model (sweep-next-edit-v2-7B, MLX 4-bit) is capable of generating meaningful completions when the inference pipeline is correctly configured. The root cause is likely in prompt construction (model conditioned to predict stop tokens), not model quality.
- The benchmark scenarios in `tests/fixtures/scenarios.json` are representative of real-world editing patterns and serve as the primary quality signal.
- The current latency baseline (~425ms p50) should improve or hold; target is p50 < 400ms.
- Three root causes are suspected (in order of likelihood): (1) prompt formatting causes the model to predict a stop token as its first token, (2) silent exception catching masks generation errors, (3) the `is_pure_insertion_above_cursor` filter incorrectly blanks valid completions. All three are in scope.
- The existing Tabby-compatible API contract must be preserved for IDE compatibility.
