# Claude Session Context — MM26 NCAAM
> **Paste this at the start of every new Claude.ai conversation.**
> Keep this file current. It's your memory layer between sessions.

---

## What We're Building
An ML model to predict NCAAM March Madness tournament outcomes.
Repo: https://github.com/billykeim/mm26_ncaam

## My Stack
- **Execution agent:** Cursor (reads the repo directly)
- **Thought partner:** Claude.ai (you — stateless, needs this file for context)
- **Source of truth:** GitHub (`DECISIONS.md` is the canonical record)

## Pipeline Protocol
1. We think + decide here in Claude.ai
2. Key decisions get written into `DECISIONS.md` and committed to GitHub
3. Cursor picks up `DECISIONS.md` + `.cursorrules` and executes
4. Results/experiments get logged back into `DECISIONS.md`
5. Next Claude session starts with this file pasted in

---

## Current Project State
> **Update this section after every session before committing.**

**Phase:** [e.g., Data Ingestion / Feature Engineering / Modeling / Evaluation]

**What's been built so far:**
- [ ] [e.g., Data pipeline pulling from KenPom]
- [ ] [e.g., Feature matrix with N features]
- [ ] [e.g., Baseline XGBoost trained on 2010–2023]

**What's currently broken or stuck:**
- [e.g., Train/test split leaking future data — need to fix]

**What we're working on this session:**
- [e.g., Deciding feature set for v1 model]

---

## Key Decisions Already Made
> Pull from DECISIONS.md — summarize the resolved ones here.

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Prediction target | [TBD] | |
| Primary data source | [TBD] | |
| Model type | [TBD] | |
| Train/test strategy | [TBD] | |

---

## Open Questions for This Session
> What do you want Claude to help you think through today?

1. [e.g., Should we use adjusted efficiency margin or raw stats?]
2. [e.g., How do we handle teams that didn't make the tournament in training years?]
3. [e.g., Review Cursor's feature engineering code — paste below]

---

## Recent Code / Output to Review
> Paste any Cursor-generated code, model outputs, or error messages here.

```
[paste here]
```
