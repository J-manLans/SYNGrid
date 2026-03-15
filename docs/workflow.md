- [**Home**](home)

# Two-Person Git Workflow for SynergyGrid
This workflow is designed to minimize merge conflicts, maintain a stable main branch, and enable parallel development.

## 1. Branch Structure
- **`main`**: always stable and tested.
- **feature/xxx**: one branch per task, method, or small feature. Example: `feature/compute_reward`, `feature/environment_step`.

## 2. Work Flow
### Step A: Start Work
1. Pull latest `main`:
```bash
git checkout main
git pull
```

2. Branch out from `main`:

```bash
git checkout -b feature/compute_reward
git checkout -b bugfix/fix_compute_reward_logic
git checkout -b hotfix/crash_on_start
git checkout -b refactor/environment_classes
# Small documentation additions can be made directly in main and pushed right away, if there is documentation concerning a feature it should be done in the feature branch
git checkout -b docs/workflow_guide
git checkout -b experiment/new_agent_arch
```

3. Work **only within the assigned method or module**. Avoid unrelated edits.

### Step B: Commit Often
* Make **small, focused commits**:

```bash
git add <files>
git commit -m "Implement basic reward calculation"
```

* Include **clear commit messages** describing what changed and where.

### Step C: Pull Latest `main` Frequently (specially if we're working in the same file)
Even if the task is not done, update your branch if the other team member have issued a PR which has been merged:

```bash
# More flexible
git fetch

git merge origin/main # Messier history
git push
```

Or:

```bash
# Replays your commits on top of main for a cleaner, linear history. You must have branched from
# main though, rebasing on a branch branched from something else can overwrite commits unexpectedly
git rebase origin/main
# Rebase rewrites history; ensures you don't overwrite commits on the branch you're working on
git push --force-with-lease
```

Not recommended:

```bash
# More straight forward, but you need to have decide upon merging strategy or set it:
# git config pull.rebase true, for example (or what strategy you choose (merge is default))
git pull origin main
```

- Resolve any conflicts immediately.
- Test locally before committing merged changes.
- Push branch updates and continue working on your branch afterward.

### Step D: Create Pull Request (PR)
- Once a logical unit (method or feature) is ready:
  1. Update one last time onto `main`.
  2. Resolve conflicts and verify functionality.
  3. Push and open PR against `main`.
- Issue review by your teammate:
  1. Check logic, style, and tests.
  2. Identify any blocking issues.
  3. Merge only after approval.

> All merges into main should be squash to maintain a clean linear history.

### Step E: Feature Merged Back Into Main
After your feature branch has been merged into `main` the other member update their branch according to step C above.

## 3. Best Practices
* **Work in isolated methods or blocks** to reduce conflicts.
* **Small, focused PRs**: Ideally, one method or feature per PR. However, since we can't dedicate full-time attention to this project, some flexibility is allowed — a PR can be slightly larger when necessary. For example, if we are working in an isolated file that doesn't affect the other team member, it's acceptable to complete the entire file before opening a PR.
* **Communicate** before modifying shared methods or files.
* **Document contracts** in code (inputs, outputs, docstring, state assumptions).
* **Lightweight review** in small teams: check for correctness and obvious issues; avoid unnecessary vetoes.
* **Pull early and often**: prevents last-minute conflicts.
* **TODO, NOTE**: communicates intent if the other team member needs to check out your branch, as well as reminding yourself about where you are and what needs to be done.