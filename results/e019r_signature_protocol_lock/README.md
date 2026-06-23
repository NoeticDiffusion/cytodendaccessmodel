# E019R — Signature Protocol Lock and Rescue Audit

**Date:** 2026-06-22

## Purpose
Lock the canonical SIG-A to SIG-E protocol before E020.

## Key outputs
- `summary/signature_definitions.md` — frozen definitions
- `summary/sig_e_rescue_audit.csv` — 5 rescue conditions
- `summary/canonical_reproduction_table.csv` — E017/E018/E019/E019R comparison
- `summary/article_signature_language.md` — locked article prose

## Key findings
1. SIG-C is an architectural fast-gating signature (not a slow-writing diagnostic)
2. SIG-D is geometry-driven, not diagnostic alone
3. SIG-E unit label corrected: normalized recovery difference (not pp)
4. SIG-E magnitude is protocol-sensitive (probe cues before rescue lift generic baseline)
5. Centralized computation in `src/cytodend_accessmodel/signatures.py`
