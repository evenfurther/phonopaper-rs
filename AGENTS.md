# Coding Agent Instructions

This file contains mandatory rules for any coding agent (AI or human) contributing
to this repository.

## Repository layout

This repository is a Cargo **workspace** with two member crates:

```
phonopaper-rs/           ← repo root (workspace)
├── Cargo.toml           ← [workspace] manifest
├── .cargo/config.toml   ← target-cpu=native
├── phonopaper-rs/       ← library crate (published to crates.io)
│   ├── Cargo.toml
│   ├── src/
│   ├── tests/           ← integration tests + fixtures
│   ├── benches/         ← Criterion and IAI-Callgrind benchmarks
│   └── examples/        ← developer / research examples
└── phonopaper-cli/      ← binary crate (cargo install phonopaper-cli)
    ├── Cargo.toml
    └── src/
        ├── main.rs      ← Cli struct, shared helpers, entry point
        └── cmd/         ← one module per subcommand
            ├── mod.rs
            ├── decode.rs
            ├── encode.rs
            ├── robust_decode.rs
            └── blank.rs
```

Workspace-level lints (`[workspace.lints.clippy] pedantic = "warn"`) are
inherited by both crates via `[lints] workspace = true` in each member's
`Cargo.toml`.

## Version control

All changes must be committed using **jujutsu (`jj`)**, not `git` directly.

- Stage and commit with `jj describe` (to set the commit message) and `jj new`
  (to start a new empty change after committing), or use `jj commit -m "…"` as a
  one-step shorthand.
- Write commit messages in the imperative mood, explaining *why* the change was
  made, not just *what* changed.
- Do not use `git commit`, `git add`, or `git push` — jujutsu manages the
  working-copy and history directly.
- After every commit, advance the `main` bookmark to the new commit:
  ```bash
  jj bookmark set main -r @-
  ```

## Required checks before finishing any task

Every change must leave the repository in a state where **all five** of the
following commands pass with zero errors and zero warnings:

```bash
# 1. Formatting — the code must be formatted exactly as rustfmt produces
cargo fmt --check --all

# 2. Lints — clippy in pedantic mode (configured in workspace Cargo.toml)
#    Clippy is run across ALL workspace members and ALL targets.
cargo clippy --workspace --all-targets

# 3. Tests — all unit tests and doc-tests must pass across the whole workspace
cargo test --workspace

# 4. Benchmarks — run the criterion benchmarks to detect heavy regressions
#    (benches live only in the phonopaper-rs library crate)
cargo bench -p phonopaper-rs --bench decode --bench encode

# 4a. IAI-Callgrind benchmarks — deterministic instruction-count measurements
#     Requires valgrind and the matching iai-callgrind-runner to be installed:
#       cargo install iai-callgrind-runner
#     Then run:
cargo bench -p phonopaper-rs --bench decode_iai --bench encode_iai

# 5. Coverage — no file may drop below its baseline (see Coverage section below)
cargo llvm-cov -p phonopaper-rs --tests --ignore-filename-regex='(benches|examples)' --summary-only
```

Run them in this order. Fix any issues before considering the task done.

> **Performance gate:** after running `cargo bench`, compare the results against
> the baseline below.  A change is acceptable if every benchmark stays within
> **±20 %** of its baseline.  Regressions beyond 20 % must be investigated and
> either justified (with a note in the commit message) or fixed before merging.
>
> Current baselines (measured on the development machine, `target-cpu=native`):
>
> | Benchmark | Baseline |
> |---|---|
> | `spectrogram_to_audio/realistic_1400col`         |  23 ms  |
> | `spectrogram_to_audio/all_bins_1400col`           | 175 ms  |
> | `synthesizer_column/single_col/realistic`         |  17 µs  |
> | `synthesizer_column/single_col/all_bins`          | 120 µs  |
> | `column_amplitudes_from_image/720px`              |   3 µs  |
> | `audio_to_spectrogram/encode_3s`                  |   9 ms  |

## Rules

### Formatting

- Always run `cargo fmt --all` after editing Rust source files.
- Do not manually wrap lines or reorder `use` statements — let `rustfmt` decide.

### Clippy (pedantic mode)

- The workspace uses `[workspace.lints.clippy] pedantic = "warn"` in the root
  `Cargo.toml`, inherited by every member crate, so every clippy warning is a
  build failure for this project's quality bar.
- **All targets are checked:** `cargo clippy --workspace --all-targets` covers
  the library, integration tests (`tests/`), benchmarks (`benches/`), examples
  (`examples/`), and the CLI binary.  Warnings anywhere are just as
  unacceptable as warnings in library code.
- Prefer fixing the root cause over silencing lints with `#[allow(...)]`.
- When a lint suppression is genuinely necessary (e.g., a deliberate lossy cast
  that has been manually proven safe), use `#[expect(clippy::lint_name,
  reason = "...")]` instead of `#[allow(...)]`. The `reason` string must explain
  **why** the suppression is safe. `#[expect]` is strictly preferred because the
  compiler will warn if the lint is later fixed, prompting cleanup.
- Never use blanket `#[allow(clippy::pedantic)]` or `#[expect(clippy::pedantic)]`
  — suppress only the specific lint that applies.

### Tests

- Do not delete or weaken existing tests.
- Follow this placement rule when adding new tests:
  - **Integration tests (`tests/`)** — tests that exercise only the public API
    (items accessible via `use phonopaper_rs::…`). This is the default location
    for any test of a `pub` function, method, or type.
  - **In-source `#[cfg(test)]` modules** — tests that must access private
    functions, types, or fields. Keep them in a `#[cfg(test)]` module at the
    bottom of the same source file as the item they test.
- Group integration tests by topic, one file per source module (e.g.
  `tests/format.rs`, `tests/spectrogram.rs`).
- Doc-examples in `///` comments must compile (`cargo test` runs them).

### Coverage

The project uses [`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov) to
measure test coverage.  The coverage baseline is tracked in this file and **must
not regress** — this check is mandatory before every commit, on the same footing
as formatting and clippy.

**Canonical command** (must match exactly so results are comparable):

```bash
cargo llvm-cov -p phonopaper-rs --tests --ignore-filename-regex='(benches|examples)' --summary-only
```

Additional useful variants:

```bash
# List every uncovered line after the summary table
cargo llvm-cov -p phonopaper-rs --tests --ignore-filename-regex='(benches|examples)' --show-missing-lines

# Interactive HTML report
cargo llvm-cov -p phonopaper-rs --tests --ignore-filename-regex='(benches|examples)' --html --open
```

**Coverage baselines:**

| File | Regions | Functions | Lines |
|---|---|---|---|
| `audio.rs`           | 87.21 % |  69.23 % |  81.68 % |
| `decode/image.rs`    | 98.39 % | 100.00 % | 100.00 % |
| `decode/markers.rs`  | 97.27 % | 100.00 % |  96.08 % |
| `decode/synth.rs`    | 96.17 % | 100.00 % |  94.74 % |
| `decode/wav.rs`      | 81.51 % |  75.00 % |  88.24 % |
| `encode.rs`          | 97.48 % | 100.00 % |  99.00 % |
| `format.rs`          |100.00 % | 100.00 % | 100.00 % |
| `render.rs`          |100.00 % | 100.00 % | 100.00 % |
| `spectrogram.rs`     | 96.08 % | 100.00 % |  97.44 % |
| `vector.rs`          | 96.13 % |  87.10 % |  91.92 % |
| **TOTAL**            | **94.84 %** | **90.82 %** | **93.71 %** |

A change is acceptable if **every file stays at or above its baseline** for all
three metrics (regions, functions, lines).  New public functions added without
accompanying tests will lower the numbers and must be caught before merging.

**Known permanently-uncovered lines** (do not attempt to cover these):

| Location | Reason |
|---|---|
| `audio.rs` — `read_mp3` `ResetRequired` branch | Signals a decoder reset at a seek point; unreachable when decoding a plain MP3 file from start to finish without seeking. |
| `audio.rs` — `read_mp3` multi-track skip branch | `packet.track_id() != track_id` guard; unreachable because `symphonia-bundle-mp3` produces a single-track stream for any valid MPEG file. |
| `decode/wav.rs:30-33` | Sample-buffer > 4 GiB overflow guard; untestable in practice. |
| `decode/synth.rs:322,578,579` | `assert_eq!` format-string arguments; only reachable on panic, not normal test flow. |
| `decode/synth.rs:428-430` | Zero-phasor reset branch in `renormalize()`; unreachable because phasors are unit-complex numbers that can only drift to zero under extreme floating-point pathology, not in normal synthesis. |
| `decode/synth.rs:609` | `synth.renormalize()` call inside `spectrogram_to_audio`; only triggered when `num_columns ≥ RENORM_INTERVAL` (128). No test synthesises that many columns; adding such a test would be expensive and the branch is covered by the direct `Synthesizer::renormalize` unit test. |
| `encode.rs:190` | Zero-padding branch that is dead code under the current frame-counting formula. |
| `decode/markers.rs:172` | Closing `)` of a multi-line `ok_or(...)` call; LLVM counts this as a separate region when `rustfmt` places it on its own line — the branch itself is tested. |
| `decode/markers.rs:193,200` | Defensive fallbacks requiring a thick stripe with no adjacent thin stripe — impossible to construct with valid `PhonoPaper` marker geometry. |
| `decode/markers.rs:204` | `data area has zero height` guard; unreachable when the defensive fallbacks above are also unreachable (the fallback values satisfy `data_bottom > data_top` by construction). |
| `vector.rs:601-603,618-620,628-629` | Error-closure bodies inside `ok_or_else(|| …)` calls in `image_from_pdf`; the corresponding `image_from_pdf_error_*` tests do exercise these paths, but LLVM counts multi-line closure bodies as separate regions and marks them uncovered when `rustfmt` spreads them across lines. |
| `vector.rs:632-638` | Size-mismatch error block in `image_from_pdf`; same LLVM multi-line closure artifact as above — the `image_from_pdf_error_size_mismatch` test exercises this path. |
| `vector.rs:647-650` | `GrayImage::from_raw` returning `None` inside `image_from_pdf`; unreachable because the `pixels.len() != img_w * img_h` guard immediately above guarantees the dimensions are consistent. |

**Coverage workflow for every task:**

1. Write tests alongside any new code — aim to cover every new public function
   and every new error branch.
2. After `cargo test` passes, run the canonical `cargo llvm-cov` command above
   and compare each file against the baseline table.
3. If any file drops below its baseline, add tests to cover the gap before
   committing.  Do not commit a coverage regression.
4. If coverage improves (numbers go up), **update the baseline table in this
   file** to record the new high-water mark.  The table must always reflect the
   actual current state of the repository, not a stale historical snapshot.
5. If a line is genuinely impossible to cover (see the table above), document
   *why* in this file, add it to the known-uncoverable table, and update the
   baseline numbers to reflect the new steady state.

### Documentation

- All public items (`pub fn`, `pub struct`, `pub enum`, `pub const`, …) must
  have a doc comment.
- Use backticks around code identifiers, type names, and proper nouns that are
  also identifiers (e.g. `` `PhonoPaper` ``, `` `SynthesisOptions` ``).
  Clippy pedantic (`doc_markdown`) enforces this automatically.
- Functions returning `Result` must have a `# Errors` section.
- Functions that can panic must have a `# Panics` section.

### Markdown files

All Markdown files at the top level of the repository (`README.md`,
`PHONOPAPER_SPEC.md`, `AGENTS.md`, …) must be kept up to date with the actual
state of the codebase.  In particular:

- **`README.md`** must accurately reflect the workspace layout, all CLI
  subcommands and their options (with correct defaults), all supported input and
  output formats, and the library's public API surface.  Update it whenever a
  subcommand is added or changed, a new format is supported, or the public API
  is extended.
- **`PHONOPAPER_SPEC.md`** must match the library's actual behaviour — default
  constants, WAV format details, supported image formats, and any known
  divergences from the reference Android application.  Update it whenever the
  relevant code changes.
- **`AGENTS.md`** (this file) must stay in sync with the project's quality gates,
  coverage baselines, and rules.  Update the coverage baseline table whenever
  coverage improves, and add entries to the known-uncoverable table whenever a
  genuinely unreachable branch is identified.

Stale documentation is treated as a defect on the same level as a failing test.
