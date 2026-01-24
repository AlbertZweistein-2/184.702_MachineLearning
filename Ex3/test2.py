from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

State = Tuple[int, int, int, int]   # (x, y, vx, vy)
Action = Tuple[int, int]            # (ax, ay) with ax,ay in {-1,0,1}


def all_actions() -> List[Action]:
    return [(ax, ay) for ax in (-1, 0, 1) for ay in (-1, 0, 1)]


@dataclass
class RaceTrackEnv:
    """
    ASCII track:
      'S' start line cells
      'F' finish/target cells
      '#' walls/obstacles
      '.' drivable track

    Coordinates:
      x = column index (0..W-1)
      y = row index    (0..H-1)
    """
    track: List[str]
    vmax: int = 2                         # since velocity components must be < 3
    allow_negative_velocity: bool = False # set True if your assignment allows negative vx/vy
    gamma: float = 1.0                    # MC discount; often 1.0 here

    def __post_init__(self) -> None:
        if not self.track:
            raise ValueError("Track must not be empty.")
        self.H = len(self.track)
        self.W = len(self.track[0])
        if any(len(r) != self.W for r in self.track):
            raise ValueError("All track rows must have equal length.")

        self.starts = [(x, y) for y in range(self.H) for x in range(self.W) if self.track[y][x] == "S"]
        self.finishes = {(x, y) for y in range(self.H) for x in range(self.W) if self.track[y][x] == "F"}

        if not self.starts:
            raise ValueError("No start cells 'S' found in track.")
        if not self.finishes:
            raise ValueError("No finish cells 'F' found in track.")

        self.reset()

    def reset(self) -> State:
        x, y = random.choice(self.starts)
        self.x, self.y = x, y
        self.vx, self.vy = 0, 0
        return (self.x, self.y, self.vx, self.vy)

    def is_wall_or_oob(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.W or y < 0 or y >= self.H:
            return True
        return self.track[y][x] == "#"

    def _clip_velocity(self, vx: int, vy: int, on_start_line: bool) -> Tuple[int, int]:
        if self.allow_negative_velocity:
            # components in [-vmax, vmax]
            vx = max(-self.vmax, min(self.vmax, vx))
            vy = max(-self.vmax, min(self.vmax, vy))
        else:
            # components in [0, vmax]
            vx = max(0, min(self.vmax, vx))
            vy = max(0, min(self.vmax, vy))

        # "cannot both be zero except at starting line"
        if vx == 0 and vy == 0 and not on_start_line:
            # minimal nudge: keep vy if possible, else vx; otherwise pick (0,1) if allowed
            if self.allow_negative_velocity:
                vy = 1
            else:
                vy = 1  # safe since nonnegative mode allows 1
        return vx, vy

    def _trace_cells(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """
        Trace cells along a move from (x0,y0) to (x1,y1), excluding the start cell.
        Simple integer interpolation based on max(|dx|,|dy|).
        """
        dx = x1 - x0
        dy = y1 - y0
        n = max(abs(dx), abs(dy))
        if n == 0:
            return []
        cells = []
        for t in range(1, n + 1):
            xt = x0 + round(dx * t / n)
            yt = y0 + round(dy * t / n)
            cells.append((xt, yt))
        # remove duplicates (can happen due to rounding)
        dedup = []
        prev = None
        for c in cells:
            if c != prev:
                dedup.append(c)
            prev = c
        return dedup

    def step(self, action: Action) -> Tuple[State, int, bool, Dict]:
        ax, ay = action
        if ax not in (-1, 0, 1) or ay not in (-1, 0, 1):
            raise ValueError("Action must be (ax,ay) with each in {-1,0,1}.")

        on_start_line = (self.track[self.y][self.x] == "S")

        # velocity update + constraints
        nvx = self.vx + ax
        nvy = self.vy + ay
        nvx, nvy = self._clip_velocity(nvx, nvy, on_start_line=on_start_line)

        # propose new position
        nx = self.x + nvx
        ny = self.y + nvy

        # trace path to detect finish/collision
        path = self._trace_cells(self.x, self.y, nx, ny)

        # reward is always -1 per step until finish (finish also ends episode)
        reward = -1

        # Check if we reach finish along the path
        for (px, py) in path:
            if (px, py) in self.finishes:
                self.x, self.y = px, py
                self.vx, self.vy = nvx, nvy
                return (self.x, self.y, self.vx, self.vy), reward, True, {"event": "finish"}

            if self.is_wall_or_oob(px, py):
                # collision: reset to random start, velocity to 0
                s = self.reset()
                return s, reward, False, {"event": "collision"}

        # If path is empty, still need to check final cell
        if not path:
            # stays in place (rare because we prevent (0,0) except start)
            pass

        # normal move
        self.x, self.y = nx, ny
        self.vx, self.vy = nvx, nvy
        return (self.x, self.y, self.vx, self.vy), reward, False, {"event": "move"}


def epsilon_greedy(Q: Dict[Tuple[State, Action], float], state: State, actions: List[Action], eps: float) -> Action:
    if random.random() < eps:
        return random.choice(actions)

    # greedy (break ties randomly)
    best_val = None
    best_actions: List[Action] = []
    for a in actions:
        v = Q.get((state, a), 0.0)
        if best_val is None or v > best_val:
            best_val = v
            best_actions = [a]
        elif v == best_val:
            best_actions.append(a)

    return random.choice(best_actions) if best_actions else random.choice(actions)


def mc_control_on_policy(
    env: RaceTrackEnv,
    num_episodes: int = 50_000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.99995,
    alpha: Optional[float] = None,
    max_steps_per_episode: int = 10_000,
    seed: Optional[int] = 0,
    progress_every: int = 1000,          # <-- NEW
) -> Tuple[Dict[Tuple[State, Action], float], Dict[Tuple[State, Action], int]]:
    if seed is not None:
        random.seed(seed)

    actions = all_actions()
    Q: Dict[Tuple[State, Action], float] = {}
    N: Dict[Tuple[State, Action], int] = {}

    eps = eps_start

    # rolling stats for progress printing
    recent_lengths: List[int] = []
    recent_finishes: List[int] = []  # 1 if finished, 0 otherwise

    for ep in range(1, num_episodes + 1):
        s = env.reset()
        episode: List[Tuple[State, Action, int]] = []
        finished = 0

        for t in range(max_steps_per_episode):
            a = epsilon_greedy(Q, s, actions, eps)
            s2, r, done, info = env.step(a)
            episode.append((s, a, r))
            s = s2
            if done:
                finished = 1
                break

        # update rolling stats
        recent_lengths.append(len(episode))
        recent_finishes.append(finished)
        if len(recent_lengths) > progress_every:
            recent_lengths.pop(0)
            recent_finishes.pop(0)

        # Monte Carlo return computation (First-visit)
        G = 0.0
        seen_sa = set()
        for (s, a, r) in reversed(episode):
            G = r + env.gamma * G
            sa = (s, a)
            if sa in seen_sa:
                continue
            seen_sa.add(sa)

            if alpha is None:
                N[sa] = N.get(sa, 0) + 1
                q_old = Q.get(sa, 0.0)
                Q[sa] = q_old + (G - q_old) / N[sa]
            else:
                q_old = Q.get(sa, 0.0)
                Q[sa] = q_old + alpha * (G - q_old)

        # epsilon schedule
        eps = max(eps_end, eps * eps_decay)

        # progress print
        if progress_every > 0 and (ep % progress_every == 0 or ep == 1):
            avg_len = sum(recent_lengths) / len(recent_lengths)
            finish_rate = 100.0 * (sum(recent_finishes) / len(recent_finishes))
            print(
                f"[episode {ep:>7}/{num_episodes}] "
                f"eps={eps:.4f}  avg_len(last {len(recent_lengths)}): {avg_len:.1f}  "
                f"finish_rate(last {len(recent_finishes)}): {finish_rate:.1f}%"
            )

    return Q, N



def greedy_rollout(env: RaceTrackEnv, Q: Dict[Tuple[State, Action], float], max_steps: int = 10_000) -> int:
    actions = all_actions()
    s = env.reset()
    for t in range(1, max_steps + 1):
        a = epsilon_greedy(Q, s, actions, eps=0.0)
        s, r, done, info = env.step(a)
        if done:
            return t
    return max_steps


if __name__ == "__main__":
    # Example tiny track (replace with your Figure 1)
    # S = start line, F = finish, # = wall, . = track
    track = [
        "###############",
        "#S....#......F#",
        "#S....#.......#",
        "#S....#.......#",
        "#S............#",
        "###############",
    ]

    env = RaceTrackEnv(track=track, vmax=2, allow_negative_velocity=False, gamma=1.0)

    Q, N = mc_control_on_policy(
        env,
        num_episodes=30_000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.9999,
        alpha=0.1,                 # constant step-size converges faster in practice
        max_steps_per_episode=5000,
        seed=0,
    )

    steps = greedy_rollout(env, Q, max_steps=5000)
    print("Greedy steps to finish (rough sanity check):", steps)
