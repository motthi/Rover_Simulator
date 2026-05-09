from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
import os
from pathlib import Path


class EpisodeInfoCallback(BaseCallback):
    """Callback that reads per-episode info from environments and logs
    success/collision rates to TensorBoard under `custom/` namespace.

    It expects the env(s) to implement `pop_episode_results()` which returns
    a list of dicts like `{'success': bool, 'collision': bool}` for
    episodes that finished since the last call.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.total_eps = 0
        self.total_success = 0
        self.total_collision = 0

    def _on_step(self) -> bool:
        # Required abstract method. We don't need per-step logic for this callback.
        return True

    def _on_rollout_end(self) -> None:
        # env_method returns a list with one element per environment
        try:
            res = self.training_env.env_method('pop_episode_results')
        except Exception:
            return True

        # res is list per env; each element is list of episode dicts
        n_eps = 0
        n_succ = 0
        n_coll = 0
        for env_res in res:
            if not env_res:
                continue
            for r in env_res:
                n_eps += 1
                if r.get('success'):
                    n_succ += 1
                if r.get('collision'):
                    n_coll += 1

        if n_eps > 0:
            self.total_eps += n_eps
            self.total_success += n_succ
            self.total_collision += n_coll

            # rolling metrics for this rollout
            roll_succ_rate = n_succ / n_eps
            roll_coll_rate = n_coll / n_eps
            # cumulative metrics
            cum_succ_rate = self.total_success / self.total_eps
            cum_coll_rate = self.total_collision / self.total_eps

            # record to tensorboard
            self.logger.record('custom/rollout_success_rate', roll_succ_rate)
            self.logger.record('custom/rollout_collision_rate', roll_coll_rate)
            self.logger.record('custom/cumulative_success_rate', cum_succ_rate)
            self.logger.record('custom/cumulative_collision_rate', cum_coll_rate)

        return True


class RolloutCheckpointCallback(BaseCallback):
    """Save model at the end of every rollout (one per _on_rollout_end call).

    Files are saved into `save_dir` with prefix `name_prefix` and suffix the rollout idx.
    """

    def __init__(self, save_dir: str = "models", name_prefix: str = "ppo_rover", verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.name_prefix = name_prefix
        self.rollout_idx = 0
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        path = os.path.join(self.save_dir, f"{self.name_prefix}_rollout_{self.rollout_idx}")
        try:
            self.model.save(path)
            if self.verbose:
                print(f"Saved rollout checkpoint: {path}.zip")
        except Exception:
            if self.verbose:
                print(f"Failed to save rollout checkpoint: {path}")
        self.rollout_idx += 1
        return True
