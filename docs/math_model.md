# Gacha-DSS Mathematical Modeling Notes

## 1. State and Process
Define the state as:
- `pity_t`: pulls since the last 5-star
- `g_t`: guarantee flag (whether the next 5-star is guaranteed to be the target)
- `c_t`: capture/target-path counter

The pulling process can be modeled as a non-stationary Markov chain:

`S_{t+1} = f(S_t, a_t, ξ_t)`

Where:
- `a_t` is the action (pull/save)
- `ξ_t` is a random variable (hit a 5-star / hit the target)

## 2. Two-Stage Probability Decomposition
The model output uses a two-stage factorization:

`P(target) = P(five_star) * P(target | five_star)`

Where:
- Stage A learns `P(five_star | state)`
- Stage B learns `P(target | five_star, state)`

## 3. Utility Function
Use Arrow-Pratt risk-averse utility:

`U(x) = (1 - exp(-a x)) / a`

When `a -> 0`, this approaches linear expected utility.

Decision score:

`Score = E[U(Reward - Cost)]`

Where `Reward` is the target-hit payoff and `Cost` is the pulling cost.

## 4. Decision Report
Output the following metrics:
- `P(five_star)`
- `P(target | five_star)`
- `P(target)`
- Multi-risk-coefficient utility curves
- Bucketed pity predictions vs. actual frequency
