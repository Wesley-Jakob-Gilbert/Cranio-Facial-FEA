# Validation team brief (weeks 1-3)

Owner: Wes Gilbert. Coordinator: Perplexity Computer. Started 2026-10-17.

## Context
- Repo: github.com/Wesley-Jakob-Gilbert/Cranio-Facial-FEA. Local clone: /home/user/workspace/Cranio-Facial-FEA
- A separate Claude Code session owns the `overhaul` branch / PR #1 (craniofea v0.2 package). DO NOT edit files under `src/`, `tests/`, `scripts/`, `configs/`, `README.md`, or `docs/*.md` outside `docs/validation/`. Report problems as findings instead.
- This team works on branch `validation-prep` (based on `origin/overhaul`). Write ONLY under `docs/validation/` (and `validation/` for scratch scripts/outputs). Do not commit or push; the coordinator commits.
- PR #1 headline: under a literature-calibrated daily-stimulus model, a 2 kPa resting tongue posture produces no density change; chewing dominates (~98 % of daily stimulus). A posture effect needs ~5e5-7e6 equivalent cycles/day.
- Goal: a preprint that is verified (code solves the math correctly) and validated against independent observations for a stated context of use. Budget is tiny (founder, unfunded): prefer free data and DIY.

## Rules
- Cite a URL or DOI for every factual claim. Mark anything unverified as UNVERIFIED.
- Separate verification (math/code) from validation (real-world data).
- Flag anything that needs Wes personally in a section titled `## Needs Wes`.
- Watch licensing: Open-Full-Jaw is reportedly CC BY-NC-SA 4.0 (non-commercial); MorphoGenix is a company.
