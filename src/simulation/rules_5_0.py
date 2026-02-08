from __future__ import annotations

from typing import Tuple
import random


def apply_five_star_rule(
    rng: random.Random,
    guarantee: bool,
    capture_enabled: bool,
    capture_counter: int,
    capture_hard: int,
    capture_prob: float,
    p_target_no_guarantee: float,
    p_target_guarantee: float,
) -> Tuple[bool, bool, int]:
    """
    Returns (is_target, guarantee_after, capture_counter_after).
    """
    # Determine target outcome based on guarantee
    if guarantee:
        is_target = rng.random() < p_target_guarantee
    else:
        is_target = rng.random() < p_target_no_guarantee

    # Apply capture mechanism if enabled and not target
    if capture_enabled and not is_target:
        capture_counter += 1
        if capture_counter >= capture_hard or rng.random() < capture_prob:
            is_target = True
            capture_counter = 0

    # Update guarantee: lose 50/50 -> guarantee next time, win -> clear
    if is_target:
        guarantee_after = False
    else:
        guarantee_after = True

    return is_target, guarantee_after, capture_counter
