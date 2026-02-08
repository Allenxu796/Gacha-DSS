from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import random

from src.models.rl_env import GachaEnv


@dataclass
class QConfig:
    episodes: int = 200
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.2
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    pity_bucket: int = 5


def _bucket(pity: int, bucket: int) -> int:
    return pity // bucket


def discretize_obs(obs: Dict[str, int], bucket: int) -> Tuple[int, int, int]:
    return (
        _bucket(obs["pity"], bucket),
        int(obs["guarantee"]),
        int(obs["capture_counter"]),
    )


def q_learn(env: GachaEnv, cfg: QConfig) -> Dict[Tuple[int, int, int, int], float]:
    q: Dict[Tuple[int, int, int, int], float] = {}

    def q_get(state, action):
        return q.get((state[0], state[1], state[2], action), 0.0)

    def q_set(state, action, value):
        q[(state[0], state[1], state[2], action)] = value

    epsilon = cfg.epsilon

    for _ in range(cfg.episodes):
        obs = env.reset()
        state = discretize_obs(obs, cfg.pity_bucket)
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                q0 = q_get(state, 0)
                q1 = q_get(state, 1)
                action = 0 if q0 >= q1 else 1

            next_obs, reward, done, _ = env.step(action)
            next_state = discretize_obs(next_obs, cfg.pity_bucket)

            best_next = max(q_get(next_state, 0), q_get(next_state, 1))
            target = reward + cfg.gamma * best_next
            new_q = q_get(state, action) + cfg.alpha * (target - q_get(state, action))
            q_set(state, action, new_q)

            state = next_state

        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

    return q


def derive_policy(q: Dict[Tuple[int, int, int, int], float]) -> Dict[Tuple[int, int, int], int]:
    policy: Dict[Tuple[int, int, int], int] = {}
    states = set((k[0], k[1], k[2]) for k in q.keys())
    for s in states:
        q0 = q.get((s[0], s[1], s[2], 0), 0.0)
        q1 = q.get((s[0], s[1], s[2], 1), 0.0)
        policy[s] = 0 if q0 >= q1 else 1
    return policy
