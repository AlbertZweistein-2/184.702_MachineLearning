import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def greedy_route(agent, env, max_steps=5000):
    """
    Runs one greedy episode (epsilon=0) and returns:
    - path_y: list of y positions
    - path_x: list of x positions
    - finished: whether target was reached
    """
    s = env.reset()
    path_y = [s[0]]
    path_x = [s[1]]

    done = False
    for _ in range(max_steps):
        a = agent.choose_action(s, eps=0.0)      # greedy
        s, r, done = env.step(a)
        path_y.append(s[0])
        path_x.append(s[1])
        if done:
            break

    return path_y, path_x, done


def plot_route(grid_map, path_y, path_x, finished, title_prefix="Greedy route"):
    """
    Plots the grid and overlays the route.
    Grid encoding: 0=track, 1=wall, 2=start, 3=finish
    """
    plt.figure(figsize=(7, 7))
    plt.imshow(grid_map, interpolation="nearest")
    plt.plot(path_x, path_y, marker="o", linewidth=2, markersize=3)
    plt.title(f"{title_prefix} | finished={finished} | steps={len(path_x)-1}")
    plt.grid(True)
    plt.gca().invert_yaxis()  # optional: makes (0,0) appear top-left like array indexing
    plt.show()


# =============================================================================
# Racetrack Environment (matches the assignment text)
# Grid encoding: 0=track, 1=wall/obstacle, 2=start line, 3=finish/target
# State: (y, x, vy, vx)
# Actions: 9 increments to velocity components: dvy, dvx in {-1,0,1}
# Reward: -1 each step until target is reached (including final move)
# Crash: reset to random start, velocity -> (0,0), episode continues
# Episode ends when reaching target cell.
# =============================================================================

State = Tuple[int, int, int, int]   # (y, x, vy, vx)
Action = int                        # 0..8 maps to (dvy, dvx)


def action_to_dv(action: Action) -> Tuple[int, int]:
    # 0..8 -> (-1,-1), (-1,0), (-1,1), (0,-1), ... , (1,1)
    dvy = (action // 3) - 1
    dvx = (action % 3) - 1
    return dvy, dvx


def all_actions() -> List[Action]:
    return list(range(9))


@dataclass
class RacetrackEnv:
    grid_map: np.ndarray
    vmax: int = 2  # because "velocity components are restricted to be less than 3" -> [-2,2]
    # If your coordinate system needs "up" to be negative y (usual for arrays), leave as +1
    # We implement y := y + vy, so to go "up" you need vy negative, which we allow.
    rng_seed: Optional[int] = 0

    def __post_init__(self):
        if self.rng_seed is not None:
            random.seed(self.rng_seed)
            np.random.seed(self.rng_seed)

        self.grid = np.array(self.grid_map, dtype=int)
        self.H, self.W = self.grid.shape

        self.start_positions = list(zip(*np.where(self.grid == 2)))  # (y,x)
        self.finish_positions = set(zip(*np.where(self.grid == 3)))

        if len(self.start_positions) == 0:
            raise ValueError("No start cells (value 2) found.")
        if len(self.finish_positions) == 0:
            raise ValueError("No finish cells (value 3) found.")

        self.reset()

    def reset(self) -> State:
        # "randomly selected start states with both velocity components zero"
        y, x = random.choice(self.start_positions)
        self.y, self.x = y, x
        self.vy, self.vx = 0, 0
        return (self.y, self.x, self.vy, self.vx)

    def _is_start_cell(self, y: int, x: int) -> bool:
        return self.grid[y, x] == 2

    def _in_bounds(self, y: int, x: int) -> bool:
        return 0 <= y < self.H and 0 <= x < self.W

    def _is_wall(self, y: int, x: int) -> bool:
        # out-of-bounds is treated as wall
        if not self._in_bounds(y, x):
            return True
        return self.grid[y, x] == 1

    def _is_finish(self, y: int, x: int) -> bool:
        return (y, x) in self.finish_positions

    def _clip_velocity(self, vy: int, vx: int) -> Tuple[int, int]:
        # We allow negative velocities because the assignment does not forbid it
        # and array coordinates often need vy<0 to go "up".
        vy = max(-self.vmax, min(self.vmax, vy))
        vx = max(-self.vmax, min(self.vmax, vx))
        return vy, vx

    def _trace_cells(self, y0: int, x0: int, y1: int, x1: int) -> List[Tuple[int, int]]:
        """
        Trace intermediate cells between start and end to detect wall collisions and finish hits.
        Uses max(|dy|,|dx|) micro-steps with rounding and de-dup.
        Excludes the start cell, includes the last cell.
        """
        dy = y1 - y0
        dx = x1 - x0
        n = max(abs(dy), abs(dx))
        if n == 0:
            return []

        cells = []
        prev = (y0, x0)
        for t in range(1, n + 1):
            yt = int(round(y0 + dy * (t / n)))
            xt = int(round(x0 + dx * (t / n)))
            cur = (yt, xt)
            if cur != prev:
                cells.append(cur)
            prev = cur
        return cells

    def step(self, action: Action) -> Tuple[State, int, bool]:
        """
        Returns: next_state, reward, done
        """
        dvy, dvx = action_to_dv(action)

        # Update velocity
        new_vy = self.vy + dvy
        new_vx = self.vx + dvx
        new_vy, new_vx = self._clip_velocity(new_vy, new_vx)

        # "cannot both be zero except at the starting line"
        on_start = self._is_start_cell(self.y, self.x)
        if (new_vy == 0 and new_vx == 0) and (not on_start):
            # simplest strict enforcement: keep previous non-zero velocity if possible
            if not (self.vy == 0 and self.vx == 0):
                new_vy, new_vx = self.vy, self.vx
            else:
                # fallback: force a minimal movement
                new_vy = -1  # usually "up" in array coords
                new_vx = 0

        # Proposed new position
        y1 = self.y + new_vy
        x1 = self.x + new_vx

        # Path check (important when velocity > 1)
        path = self._trace_cells(self.y, self.x, y1, x1)

        reward = -1  # "-1 for each step until the robot reaches the target"

        # Traverse path: first check finish, then wall (or vice versa)?
        # Typical racetrack: hitting finish anywhere along path ends episode.
        for (yy, xx) in path:
            if self._is_finish(yy, xx):
                # Reached target: episode ends
                self.y, self.x = yy, xx
                self.vy, self.vx = new_vy, new_vx
                return (self.y, self.x, self.vy, self.vx), reward, True

            if self._is_wall(yy, xx):
                # Crash: move back to random start, velocity to zero, episode continues
                s = self.reset()
                return s, reward, False

        # If path empty (should only happen if forced zero at start), still handle finish/wall at end cell
        if not path:
            if self._is_finish(y1, x1):
                self.y, self.x = y1, x1
                self.vy, self.vx = new_vy, new_vx
                return (self.y, self.x, self.vy, self.vx), reward, True
            if self._is_wall(y1, x1):
                s = self.reset()
                return s, reward, False

        # Normal move
        self.y, self.x = y1, x1
        self.vy, self.vx = new_vy, new_vx
        return (self.y, self.x, self.vy, self.vx), reward, False


# =============================================================================
# On-policy First-Visit Monte Carlo Control (epsilon-greedy)
# =============================================================================

class MonteCarloControlAgent:
    def __init__(
        self,
        env: RacetrackEnv,
        gamma: float = 1.0,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.9999,
        alpha: Optional[float] = 0.1,  # if None -> incremental mean via counts
        max_steps_per_episode: int = 5000,
    ):
        self.env = env
        self.gamma = gamma

        self.eps = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.alpha = alpha
        self.max_steps = max_steps_per_episode

        # Q: state -> np.array(9)
        self.Q: Dict[State, np.ndarray] = {}
        # counts for incremental mean if alpha is None
        self.N: Dict[Tuple[State, Action], int] = {}

        self.actions = all_actions()

    def _qvals(self, s: State) -> np.ndarray:
        if s not in self.Q:
            self.Q[s] = np.zeros(9, dtype=float)
        return self.Q[s]

    def choose_action(self, s: State, eps: Optional[float] = None) -> Action:
        if eps is None:
            eps = self.eps

        if random.random() < eps:
            return random.choice(self.actions)

        q = self._qvals(s)
        mx = np.max(q)
        best = np.where(q == mx)[0]
        return int(random.choice(best))

    def generate_episode(self) -> Tuple[List[Tuple[State, Action, int]], bool]:
        episode: List[Tuple[State, Action, int]] = []
        s = self.env.reset()
        done = False

        for _ in range(self.max_steps):
            a = self.choose_action(s)
            s2, r, done = self.env.step(a)
            episode.append((s, a, r))
            s = s2
            if done:
                break

        return episode, done

    def update_first_visit(self, episode: List[Tuple[State, Action, int]]) -> None:
        G = 0.0
        seen = set()

        for (s, a, r) in reversed(episode):
            G = r + self.gamma * G
            sa = (s, a)
            if sa in seen:
                continue
            seen.add(sa)

            q = self._qvals(s)
            if self.alpha is None:
                self.N[sa] = self.N.get(sa, 0) + 1
                n = self.N[sa]
                q[a] = q[a] + (G - q[a]) / n
            else:
                q[a] = q[a] + self.alpha * (G - q[a])

    def decay_epsilon(self) -> None:
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def greedy_eval(self, runs: int = 10, max_steps: Optional[int] = None) -> Tuple[float, float]:
        """
        Returns: (success_rate, avg_steps_success)
        """
        if max_steps is None:
            max_steps = self.max_steps

        successes = 0
        steps_success = []

        for _ in range(runs):
            s = self.env.reset()
            done = False
            for t in range(1, max_steps + 1):
                a = self.choose_action(s, eps=0.0)
                s, r, done = self.env.step(a)
                if done:
                    successes += 1
                    steps_success.append(t)
                    break

        success_rate = successes / runs
        avg_steps = float(np.mean(steps_success)) if steps_success else float("inf")
        return success_rate, avg_steps


def train(
    env: RacetrackEnv,
    num_episodes: int = 50_000,
    progress_every: int = 1000,
    eval_runs: int = 10,
) -> MonteCarloControlAgent:
    agent = MonteCarloControlAgent(
        env,
        gamma=1.0,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.9999,
        alpha=0.1,                  # constant step-size often works well in practice
        max_steps_per_episode=5000,
    )

    recent_len: List[int] = []
    recent_finish: List[int] = []

    for ep in range(1, num_episodes + 1):
        episode, finished = agent.generate_episode()
        agent.update_first_visit(episode)
        agent.decay_epsilon()

        recent_len.append(len(episode))
        recent_finish.append(1 if finished else 0)
        if len(recent_len) > progress_every:
            recent_len.pop(0)
            recent_finish.pop(0)

        if ep == 1 or (progress_every > 0 and ep % progress_every == 0):
            avg_len = sum(recent_len) / len(recent_len)
            finish_rate = 100.0 * (sum(recent_finish) / len(recent_finish))

            # Greedy eval to see if it REALLY learned something
            sr, avg_steps = agent.greedy_eval(runs=eval_runs)
            sr_pct = 100.0 * sr
            avg_steps_str = f"{avg_steps:.1f}" if np.isfinite(avg_steps) else "inf"

            print(
                f"[ep {ep:>6}/{num_episodes}] "
                f"eps={agent.eps:.4f}  "
                f"train_finish(last {len(recent_finish)}): {finish_rate:5.1f}%  "
                f"avg_len(last {len(recent_len)}): {avg_len:7.1f}  "
                f"GREEDY eval: {sr_pct:5.1f}% success, avg_steps={avg_steps_str}"
            )

    return agent


# =============================================================================
# Example usage
# Replace grid_map with your real racetrack map from Figure 1.
# =============================================================================
if __name__ == "__main__":
    # Example mini-track:
    # 1=wall, 0=track, 2=start line (bottom), 3=finish (top-right-ish)
    grid_map = np.array([
        [1,1,1,1,1,1,1,1,1],
        [1,0,0,3,0,0,0,0,1],
        [1,0,1,1,1,1,0,0,1],
        [1,0,0,0,0,1,0,0,1],
        [1,0,1,0,0,1,0,0,1],
        [1,0,1,0,0,0,0,0,1],
        [1,2,2,2,2,2,2,2,1],
        [1,1,1,1,1,1,1,1,1],
    ], dtype=int)

    env = RacetrackEnv(grid_map, vmax=2, rng_seed=0)
    agent = train(env, num_episodes=30000, progress_every=1000, eval_runs=10)
    
    # After training:
    path_y, path_x, finished = greedy_route(agent, env, max_steps=5000)
    plot_route(env.grid, path_y, path_x, finished)


    # Final greedy test:
    sr, avg_steps = agent.greedy_eval(runs=30)
    print(f"\nFinal GREEDY success rate: {100*sr:.1f}%, avg steps (success only): {avg_steps:.1f}")
