# Git bootstrap and source-naming policy

Current state: `~/projects/EGO-Det` has already been initialized as an empty local `main` Git repository; the first commit failed because author identity was missing. **Do not run `git init` again.**

## A. Protect author identity (only this repository)

Go to https://github.com/settings/emails and copy the **actual** GitHub-provided `noreply` email address (often `ID+USERNAME@users.noreply.github.com`). Do not guess the numeric prefix. Enable GitHub email privacy and blocking pushes that expose your private email address.

```bash
cd ~/projects/EGO-Det
git config --local user.name "NullParallax"
git config --local user.email "PASTE_YOUR_ACTUAL_GITHUB_NOREPLY_ADDRESS"
git config --local init.defaultBranch main
# Check what the very next commit will disclose:
git var GIT_AUTHOR_IDENT
git var GIT_COMMITTER_IDENT
```

**Never commit with the placeholder email.** Use the real `noreply` address from the Settings page. This command is repository-local and will not modify identities for ODTrack/OSTrack repositories.

## B. Save the pristine imported baseline FIRST (before applying the companion patch)

```bash
cd ~/projects/EGO-Det
git status --short
git add -A
git diff --cached --name-only
git diff --cached --check
# The following should print nothing; if it prints paths, review before committing:
git diff --cached --name-only | grep -E '(^|/)(data|outputs|weights|checkpoints|\.idea|\.env)(/|$)' || true
# Check for accidentally staged large files:
git ls-files -z | xargs -0 -r du -ah | sort -h | tail -n 10
# Check staged text for common key markers; inspect any matches without printing secrets publicly:
git grep --cached -l -I -E 'OPENAI_API_KEY|ghp_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+' || true
# Stop and audit if any sensitive material is detected.
git commit -m "chore: import EGO-Det V2 source baseline"
```

The first commit intentionally preserves the original V2 names and mathematics. **Commit only after confirming the actual email address**, paths and licenses; it is local, not a GitHub push.

## C. Apply the patch and make a SECOND commit

After recording the baseline, place the phase-1 patch zip into the Windows project folder, then:

```bash
cd ~/projects/EGO-Det
unzip -o '/mnt/g/EgoNet5.0/EGO-Det_git_env_phase1_patch.zip' -d .
git status --short
git diff -- .gitignore README.md requirements.txt
git add .gitignore .gitattributes README.md environment.yml requirements.txt requirements-dev.txt docs/ENVIRONMENT.md docs/GIT_SETUP_AND_NAMING.md docs/NAMING_MIGRATION_PLAN.md
git diff --cached --stat
git commit -m "build: define minimal EGO-Det environment and repository policy"
git log --oneline --max-count=2
```

Before committing, `README.md`, `requirements.txt` and `.gitignore` should match the patch files. `docs/staging_file_manifest.json` is an **import-time provenance snapshot**, not a live list of current file hashes.

## D. Create a private GitHub remote only after review

On https://github.com/new, create `EGO-Det` under the correct account; choose **Private**; do not auto-initialize README, license, or `.gitignore`.

After login/authentication is set up and after any secrets/license review:

```bash
cd ~/projects/EGO-Det
git remote -v
# If no remote is present, add the exact URL displayed by GitHub:
git remote add origin https://github.com/NullParallax/EGO-Det.git
git remote -v
git push -u origin main
```

If the actual account spelling differs, use the real URL from GitHub. HTTPS authentication may require Git Credential Manager, GitHub CLI, or a personal access token (never paste tokens into shell history or Git remote URLs). Private does not mean safe for secrets; never push an API key. Do not make the repository Public until checkpoint, licensing, anonymous-review, and permission checks have passed.

## E. Commit cadence

- `chore: import EGO-Det V2 source baseline`
- `build: define minimal environment and repository policy`
- `refactor: consolidate compatibility paths` (only after tests and a source diff review)
- `test: verify strict checkpoint loading`
- `test: reproduce COCO val2017 metrics`
- `docs: publish model documentation and citation`

Keep each commit narrow; do not combine a rename with numerical/architectural changes.
