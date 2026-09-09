# Recording a Fitness-Based Checkpoint — Step-by-Step Guide

Right now, the checkpoint warm-starting resumes from is selected by rl_games'
own notion of "best" — the epoch with the highest **raw training reward**, not
the epoch with the highest **`fitness_function`** (the actual ground-truth
metric ARD uses to judge every candidate). This guide covers what's needed to
change that: a new save trigger in the forked rl_games, and a small selection
change in ARD. No code has been written yet — this documents the approach.

## Why this isn't already the case

`fitness_function` isn't computed or known by rl_games at all — it's written
by each task's own env code (`self.extras["log"]["fitness_function"] = ...`,
e.g. in `shadow_hand_env.py`/`cartpole_env.py`) and only reaches TensorBoard
via `IsaacAlgoObserver`, which generically forwards anything in `infos`/`log`
to the writer. rl_games' actual checkpoint-saving logic
(`rl_games/common/a2c_common.py`) only ever compares against its own tracked
**raw reward** — it has no path to `fitness_function` at all today.

## Where this repo's fork already lives

Confirmed on disk: `/home/harsh/Documents/Github/rl_games` (remote
`git@github.com:UoA-CARES/rl_games.git`, current branch `Main`, tracking the
same commit as the fork's `master`). `ard-isaaclab-tasks`' Dockerfile (on
`feature/plasticity-configs`, where the plasticity work already lives) already
installs **this fork**, not stock rl_games — `RL_GAMES_REPO`/`RL_GAMES_REF`
build args, defaulting to `master`. So `ard-isaaclab-tasks` needs **no
changes** for this — it's already wired to pull from the fork; the actual work
is entirely inside the fork itself.

Also worth knowing: a branch named `feat-able-to-restore-transfer-specific-states`
already exists on the fork, at the same commit as `master` — this is very
likely Futian's "transfer" mechanism work from the earlier Slack discussion,
already in progress. Worth coordinating with him before starting this, since
both changes touch adjacent territory (checkpoint save/load behavior).

## Step 1 — Add a fitness tracker + save trigger to `IsaacAlgoObserver`

**File**: `rl_games/common/algo_observer.py` (in the fork), class `IsaacAlgoObserver`.

Confirmed directly from this file: `after_init(self, algo)` stores
`self.algo = algo` — the observer already holds a **live reference to the
agent itself**. This is the cleanest integration point: the observer can
call `self.algo.save(...)` directly, with zero changes needed to
`a2c_common.py`'s core training loop.

Mirror the existing raw-reward pattern (`a2c_common.py`'s
`self.last_mean_rewards` / `save_best_after` / `self.save(...)`), but scoped
entirely inside this class:

```python
class IsaacAlgoObserver(AlgoObserver):
    def after_init(self, algo):
        self.algo = algo
        self.mean_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.ep_infos = []
        self.direct_info = {}
        self.writer = self.algo.writer
        # New: track fitness_function the same way a2c_common.py tracks reward.
        self.fitness_tag = "fitness_function"
        self.mean_fitness = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.last_mean_fitness = -1_000_000_000

    def process_infos(self, infos, done_indices):
        ...  # existing body, unchanged
        # New: infos["log"][self.fitness_tag] carries the per-step value the
        # task env writes (see cartpole_env.py / shadow_hand_env.py). Update
        # the tracker only for envs whose episode just ended, mirroring how
        # a2c_common.py's game_rewards.update() is gated on done_indices.
        log = infos.get("log", {})
        if self.fitness_tag in log and len(done_indices) > 0:
            value = log[self.fitness_tag]
            self.mean_fitness.update(value[done_indices] if hasattr(value, "__getitem__") else value)

    def after_print_stats(self, frame, epoch_num, total_time):
        ...  # existing body, unchanged
        # New: mirror a2c_common.py's own best-checkpoint save trigger, but
        # keyed on fitness_function instead of raw reward, and saved to a
        # DIFFERENT filename so it never collides with/overwrites the
        # existing reward-best file — both can coexist.
        if self.mean_fitness.current_size > 0:
            mean_fitness = self.mean_fitness.get_mean()
            self.writer.add_scalar("fitness/mean", mean_fitness, frame)
            if (
                mean_fitness > self.last_mean_fitness
                and epoch_num >= self.algo.save_best_after
            ):
                self.last_mean_fitness = mean_fitness
                self.algo.save(
                    os.path.join(self.algo.nn_dir, self.algo.config["name"] + "_fitness")
                )
```

**Before implementing this exactly as written**: confirm the precise shape of
`infos["log"]` at the point `process_infos` receives it (is `fitness_function`
already a per-env tensor indexable by `done_indices`, or a single aggregate
scalar the env already reduced with `.mean()`? Both `cartpole_env.py` and
`shadow_hand_env.py` write `.mean()`-reduced scalars into `extras["log"]`,
which means the value arriving here may already be a single number across all
envs, not per-env — in that case, update the tracker on every `after_print_stats`
call using the latest logged scalar directly, rather than trying to index by
`done_indices` the way raw reward does. Trace `direct_info`'s existing handling
in the same file (visible further down, not shown above) to confirm exactly
how already-reduced scalars like this get carried from `process_infos` to
`after_print_stats` today, and follow that same path for fitness.

## Step 2 — Confirm/update the Dockerfile's `RL_GAMES_REF`

**File**: `ard-isaaclab-tasks/Dockerfile` (on `feature/plasticity-configs`).

Already defaults to tracking `master`, which already matches the fork's
current tip. Once Step 1 is merged into `master` (or whichever branch you
land it on), a plain rebuild picks it up automatically — no Dockerfile edit
needed unless you want to pin `RL_GAMES_REF` to an exact commit for
reproducibility (recommended once this stabilizes, same reasoning as the
`rl_games==1.6.1` pin discussed for the plasticity work).

## Step 3 — Prefer the new file in ARD's checkpoint selection

**File**: `src/evaluation/result_processor.py`, `find_checkpoint()`.

Currently prefers the plain `<name>.pth` (reward-best) over `last_*.pth`
(periodic) files. Once Step 1 lands, a third file exists:
`<name>_fitness.pth`. Prefer it, when present, over the reward-best file —
falling back to the existing reward-best logic if it doesn't exist yet (e.g.
a run trained before this change, or on a codebase not yet updated):

```python
@staticmethod
def find_checkpoint(run_dir: str) -> Optional[str]:
    candidates = glob.glob(os.path.join(run_dir, "nn", "*.pth"))
    if not candidates:
        return None
    fitness_best = [c for c in candidates if os.path.basename(c).endswith("_fitness.pth")]
    if fitness_best:
        return max(fitness_best, key=os.path.getmtime)
    best = [c for c in candidates if not os.path.basename(c).startswith("last_")]
    if best:
        return max(best, key=os.path.getmtime)
    return max(candidates, key=os.path.getmtime)
```

No changes needed anywhere else in ARD — everything downstream (baking into
the tarball, `--checkpoint` delivery, `_effective_max_iterations`) operates
on whatever path `find_checkpoint()` hands back, regardless of which file it
came from.

## Testing / verification plan

Mirror the methodology already used for warm-start and plasticity in this
project:

1. Run a task (cartpole first, fast) with the updated fork built in.
2. Confirm both files now exist side by side in a candidate's `nn/` folder:
   `<name>.pth` (reward-best) and `<name>_fitness.pth` (fitness-best).
3. Load both with `torch.load` (same technique used earlier this session) and
   compare their `epoch` fields — on a run where raw reward and
   `fitness_function` diverge, these should differ, confirming the two
   triggers are genuinely tracking different things, not coincidentally
   always saving at the same epoch.
4. Confirm `ResultProcessor.find_checkpoint()` returns the `_fitness.pth` path
   once present.
5. Re-run a warm-start comparison (same shape as the earlier cartpole/shadow_hand
   tests) — once using the old reward-best selection, once using the new
   fitness-best selection — and compare `fitness_function` at equal epoch
   budgets in the next iteration, the same way warm-start-vs-cold-start was
   compared earlier.

## Open items to resolve before implementing

- Exact shape of `infos["log"]["fitness_function"]` inside `process_infos`
  (per-env tensor vs. already-reduced scalar) — trace `direct_info`'s
  existing handling in `algo_observer.py` to confirm.
- Coordinate with Futian on the `feat-able-to-restore-transfer-specific-states`
  branch, since both changes touch checkpoint save/load behavior on the same
  fork.
- Decide whether `save_best_after`'s existing grace period (shared with the
  reward-best trigger) is the right gate for fitness too, or whether it
  deserves its own separately-configured threshold.
