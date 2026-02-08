from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import random

from src.simulation.engine import GachaEngine, SimulationConfig, State


@dataclass
class EnvConfig:
    max_steps: int = 365
    pull_cost: float = 1.0
    reward_target: float = 10.0
    reward_five_star: float = 2.0


class GachaEnv:
    """
    Minimal Gym-like environment.
    Action: 0 = save, 1 = pull
    Observation: dict with current pity/guarantee/capture
    Reward: weighted by target/5-star outcomes minus cost
    """

    def __init__(self, sim_config: SimulationConfig, env_config: EnvConfig):
        self.sim_config = sim_config
        self.env_config = env_config
        self.engine = GachaEngine(sim_config)
        self.reset()

    def _obs(self) -> Dict[str, int]:
        s = self.engine.state
        return {
            "pity": s.pulls_since_five_star,
            "guarantee": int(s.guarantee),
            "capture_counter": s.capture_counter,
            "steps": self.steps,
        }

    def reset(self) -> Dict[str, int]:
        self.engine.reset()
        self.steps = 0
        return self._obs()

    def step(self, action: int) -> Tuple[Dict[str, int], float, bool, Dict]:
        if action not in (0, 1):
            raise ValueError("action must be 0 (save) or 1 (pull)")

        self.steps += 1
        done = self.steps >= self.env_config.max_steps
        reward = 0.0

        if action == 1:
            result = self.engine.pull_once()
            reward -= self.env_config.pull_cost
            if result.is_five_star:
                reward += self.env_config.reward_five_star
            if result.is_target:
                reward += self.env_config.reward_target
        else:
            # save action: no pull, no cost, small time penalty optional
            reward += 0.0

        return self._obs(), reward, done, {}
