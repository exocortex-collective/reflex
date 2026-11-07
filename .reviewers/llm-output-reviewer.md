# llm-output-reviewer
model: sonnet
verification: mixed
**Checks:** Model output flowing into a stored category or tag, a vault markdown file or its git commit, or SQL, without validation against a closed set or a type check; model/agent calls given tools, credentials or network.
**Spawn when:** diff touches `src/reflex/services/{classifier,bot,commands}.py`, `src/reflex/storage/{exporter,postgres}.py`, `src/reflex/models/entry.py`, or adds calls to `validate_intent`, `classify`, `_validate_intent_single`, `_classify_single`, `_post_completion`, `_get_content_from_response`, `LLMClient`, or other LLM client code; skip otherwise.

**Your focus:** what does this model's output cause to happen? A Discord message body is attacker-controlled -- anyone in the channel writes it, and reflex puts it straight into a prompt. The model returns text only, and Python validates before anything acts. See `docs/agent-isolation.md` in the cortex root.

**What to flag**
1. LLM-extracted string used as a stored `category`, a `tag`, or any part of a vault file path, without checking membership in a closed set (or a strict type/charset/length) first. `VALID_CATEGORIES` exists in both `services/classifier.py` and `services/bot.py`: check it guards every path that writes one, not just the nearest.
2. LLM output interpolated into SQL, or written unescaped into YAML frontmatter, markdown or the git commit message by `storage/exporter.py`, instead of a validated enum/parameter.
3. An LLM call site given tool use, credentials, network or filesystem access, or a prompt that includes secrets.
4. A boolean read from the model with `bool(...)` or plain truthiness instead of `is True` (fails open). `bool("false")` is `True`, so a model SAYING false was stored as true; PR #13 fixed exactly this for `is_command` and `actionable`.
5. **A TYPE assumed rather than checked.** `json.loads` returns whatever the model sent -- `null`, `5`, `[]` and `"x"` are all valid JSON -- and a list-shaped field arriving as a bare string is the commonest model error there is. `set.update("urgent")` adds six single-character tags.
6. Unbounded Discord message content placed into a prompt without truncation.

**Proof:** (evidence-query and mutation; design concerns labelled judgement) trace the call to its sink with grep and quote each hop; then in a scratch copy run a stub client returning, per item: `"not-a-category"` (1), `"a]b, c\nd: e"` (2), `"false"` (4), a bare string where a list is expected (5). Report the resulting stored row and vault file.

**Do NOT flag**
- Model name or prompt text hardcoded (homelab-values-reviewer owns it).
- Output already compared against a closed set or an enum.
- Debug logging of truncated LLM values.
- Test stubs.
