# Security Policy

wmx-suite is a small, community-maintained tool. Security reports are taken seriously
and handled on a best-effort basis.

## Supported versions

The project is pre-1.0 and moves on `main`. Only the **latest `main`** is supported —
please confirm an issue reproduces there before reporting.

## Reporting a vulnerability

**Please do not open a public issue for a security vulnerability.**

Use GitHub's private vulnerability reporting:

1. Go to the repo's **[Security tab](https://github.com/willsarg/wmx-suite/security)**.
2. Click **"Report a vulnerability"**.
3. Describe the issue, the impact, and steps to reproduce.

This keeps the report private until a fix is available. You can expect an initial
response within about a week. If accepted, we'll work on a fix and coordinate disclosure;
if declined, we'll explain why.

## Scope

wmx-suite runs locally and has no network service, no authentication, and no remote
attack surface of its own. The areas most relevant to security are:

- **Untrusted model repositories.** The suite reads each model's `config.json` from the
  Hugging Face cache and launches `mlx_lm.generate` with passthrough arguments. A
  maliciously crafted model config or repo is a plausible vector — reports here are
  in scope.
- **Argument / path handling.** Issues in how cached models are discovered or how
  passthrough args are forwarded to `mlx_lm`.
- **Dependency vulnerabilities** that materially affect wmx-suite.

### Not a security issue

A bug where the memory predictor is *wrong* — under-predicting and risking a crash — is a
serious **safety** bug, but it isn't a vulnerability (no adversary). Please report those
through the normal **🐞 Bug report** issue form, with your `wmx-suite system` output and
the `--dry-run` plan. The project's prime directive is *never crash a machine*, so these
are still treated as high priority.

## Trademarks

MLX, Apple Silicon, Metal, Mac, and macOS are trademarks of Apple Inc. This is an
independent project, not affiliated with Apple.
