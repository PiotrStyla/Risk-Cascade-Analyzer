# Release Checklist (Retro Cascade)

This checklist is for preparing a clean, reproducible release of the Retro Cascade project.

## 0) Pre-flight (repo hygiene)

- [ ] Confirm you are on the correct branch:
  - `git branch --show-current` should be `main`
- [ ] Ensure working tree is clean:
  - `git status`
- [ ] Verify no secrets or local-only files are included:
  - `.env` should **not** be committed
  - `Future Retrospective Cascade.txt` should remain ignored

## 1) Environment setup (Windows PowerShell)

- [ ] Create and activate venv (if not already):
  - `python -m venv venv`
  - `./venv/Scripts/Activate.ps1`
- [ ] Install dependencies:
  - `pip install -r requirements.txt`

## 2) Smoke tests (must pass)

- [ ] Run CLI examples:
  - `python examples/basic_usage.py`
  - Expect: ends with `✅ All examples completed!`

- [ ] Run Streamlit UI:
  - `streamlit run app.py`
  - Verify in browser:
    - scenario selection works
    - Monte Carlo simulation runs
    - sensitivity analysis runs without errors
    - intervention comparison renders results

## 3) Versioning

Choose a versioning scheme and be consistent (recommended: SemVer).

- [ ] Decide version: `vMAJOR.MINOR.PATCH` (e.g. `v0.1.0`)
- [ ] Update documentation that references the release (if needed):
  - `README.md`
  - `QUICKSTART.md`

## 4) Final checks

- [ ] Confirm `requirements.txt` is accurate (no missing runtime deps)
- [ ] Confirm project runs from a fresh checkout:
  - `pip install -r requirements.txt`
  - `python examples/basic_usage.py`

## 5) Commit

- [ ] Create a release prep commit (if there are changes):
  - `git add -A`
  - `git commit -m "Prepare release vX.Y.Z"`

## 6) Tag and push

- [ ] Create annotated tag:
  - `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
- [ ] Push commits and tags:
  - `git push`
  - `git push --tags`

## 7) GitHub Release

On GitHub (web UI):

- [ ] Draft a new Release from tag `vX.Y.Z`
- [ ] Add release notes:
  - highlights
  - breaking changes (if any)
  - upgrade notes
- [ ] Attach assets if relevant (optional):
  - screenshots
  - sample outputs

## 8) Post-release verification

- [ ] On another machine (or fresh folder), clone the repo and re-run:
  - `pip install -r requirements.txt`
  - `python examples/basic_usage.py`
  - `streamlit run app.py`

## 9) Rollback plan (if something goes wrong)

- [ ] Revert the problematic commit:
  - `git revert <commit_sha>`
- [ ] Push the revert:
  - `git push`
- [ ] If release tag is wrong:
  - create a new patch release tag (recommended) instead of deleting public tags
