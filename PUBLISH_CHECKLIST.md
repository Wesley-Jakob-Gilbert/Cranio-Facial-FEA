# Publish Checklist (cranio_FEA)

## Pre-publish
- [x] Keep source code/config/docs in repo
- [x] Exclude generated result artifacts via `.gitignore`
- [x] Keep reproducible pipeline script (`scripts/reproduce_all.sh`)
- [x] Keep requirements file (`requirements.txt`)
- [x] Include roadmap/spec docs under `docs/`

## First push steps
1. `cd cranio_FEA`
2. `git init`
3. `git add .`
4. `git commit -m "feat: initial cranio_FEA MVP + template geometry + load scenarios"`
5. `gh repo create <your-username>/cranio_FEA --private --source=. --remote=origin --push`
   - replace `--private` with `--public` if desired

## Reproduce after clone
```bash
python3 -m pip install -r requirements.txt
./scripts/reproduce_all.sh
```
