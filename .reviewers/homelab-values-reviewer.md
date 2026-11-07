# homelab-values-reviewer
**Checks:** Hardcoded homelab IPs/hostnames, env defaults embedding them, and hardcoded LLM model names or prompts in non-test `src/` code.
**Spawn when:** diff changes non-test `*.py` under `src/reflex/` (esp. `services/classifier.py`, `services/bot.py`, `storage/exporter.py`) or adds `os.getenv`/`os.environ` defaults; skip tests-only and docs-only diffs.
**Contents:** What to flag: Homelab IPs; Homelab hostnames; Env defaults that embed homelab values; Hardcoded LLM model names; Hardcoded prompts · Review approach · Do NOT flag

Enforce CLAUDE.md's **no hardcoded homelab values** and **no hardcoded LLM models/prompts** rules. These repos are public; personal infrastructure details and un-configurable model choices must not leak into code.

**Default severity: P2** (P1 if it's a credential or a `deny`-listed secret).

**What to flag:**

1. **Homelab IPs** — string literals matching `10.5.2.\d+` (or other private-range homelab addresses) anywhere in `src/`.
2. **Homelab hostnames** — `ares`, `hades`, `gaia`, `macbook`, the `.evilsoft` domain, `ares.evilsoft`, etc., as literals in code.
3. **Env defaults that embed homelab values** — e.g. `os.getenv("OLLAMA_URL", "http://10.5.2.31:11434")`. The CLAUDE.md rule is explicit: if a service URL is required, make it *required* (raise if unset) — don't fall back to a value that exposes the internal network.
4. **Hardcoded LLM model names** — literals like `"gemma3n:e4b"`, `"phi3:mini"`, `"gpt-4..."`, `"claude-..."`, `"llama..."` passed to an LLM call. Model choice must come from an env var; reflex uses `REFLEX_LLM_TIER1_MODEL` and `REFLEX_LLM_TIER2_MODEL`, and both are required rather than defaulted.
5. **Hardcoded prompts** — a multi-line/templated prompt string literal passed into an LLM call, rather than loaded from an env var or a prompt file at startup.

**Review approach:**

1. Grep the non-test `src/` diff for `10\.5\.2\.`, `evilsoft`, `\b(ares|hades|gaia)\b`, and `os.getenv(...,` defaults containing a URL/IP.
2. Grep for likely model tokens (`:mini`, `:e4b`, `gpt-`, `claude-`, `gemma`, `llama`, `phi3`) in code that reaches an LLM client.
3. For any LLM call, confirm `model=` and `prompt=` are parameters/config lookups, not literals.

**Do NOT flag:**

- **Deployment configs** (`~/HomeLab/`, compose files, Portainer stack env) — homelab values *belong* there, not in code.
- **Tests and fixtures** using example/RFC-5737 IPs or obviously-fake hosts (`example.com`, `db`, `localhost`).
- **Docs / CLAUDE.md** that reference the real infra intentionally.
- **Prompt text held outside code** — an env var, or a file loaded at startup the way cortex-school keeps `bin/prompts/*`. That is the *correct* place for it.
- Note reflex has **no** rules-config database and no `rules.yaml`, so the triage-style "it belongs in the YAML" exemption does not apply here.
- The module-constant prompts already in `services/classifier.py` are tracked by **reflex-2f5b**. Run `bd show reflex-2f5b` before flagging them: while it is open they are known debt, and if it is closed while they remain, that is a finding. Flag any NEW hardcoded prompt regardless, and never cite those two as precedent.
- Env reads with **no** homelab default (`os.environ["OLLAMA_URL"]` or `os.getenv("OLLAMA_URL")` returning None) — those are compliant.

---
