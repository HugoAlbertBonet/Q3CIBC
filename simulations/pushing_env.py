"""Simulated Pushing environment, after the IBC paper's BlockPush task.

Reference: Florence et al., 2021, "Implicit Behavioral Cloning" (arXiv:2109.00137),
§5 / Fig. 4 — "Simulated Pushing": an effector (xArm in the paper) pushes a single
block to a single target. We re-implement the *task* (not the MuJoCo dynamics)
as a lightweight n-D environment with a PD-controlled effector and a soft-contact
block, mirroring the existing `ParticleEnv` structure so the rest of the
Q3CIBC pipeline can be reused.

Differences from `ParticleEnv`:
    * State adds (pos_block, vel_block) — observation is 6 * n_dim (no velocity-hide
      knob: the policy needs block state to push it).
    * Dynamics adds a soft contact: when |pos_agent - pos_block| < contact_radius,
      a spring force pushes the block; the block also has Coulomb-style linear
      friction so it doesn't drift after release.
    * Reward / success: the BLOCK must reach (and stay near) the target — not
      the agent. This is the non-trivial part: a near-expert agent action that
      sends the agent through the goal without pushing the block is wrong.

Observation (6 * n_dim):
    [pos_agent, vel_agent, pos_block, vel_block, pos_target, 0]   # last n_dim
    The final n_dim slot is a zero pad reserved for "secondary goal" parity
    with the particle obs layout, kept so frame-stacking math (4N → 6N) stays
    consistent for the bounds JSON. (We DO use the slot — it carries the
    target again to keep the obs shape divisible by n_dim.)
    Actually: to keep things clean we use obs_dim = 6 * n_dim with semantic
    layout [pos_a, vel_a, pos_b, vel_b, pos_target, pos_target_dup] — the
    duplicate is intentional so any downstream code that hardcoded "4 fields
    of n_dim" gets a 6-field analogue without surprises.

Action (n_dim):
    Position setpoint for the agent in [0, 1]^n.
"""

from __future__ import annotations

import copy
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces


class PushingEnv(gym.Env):
    """1-block / 1-target pushing env, n-D configurable.

    The IBC paper showed Simulated Pushing because the optimal policy is
    multi-modal: there are many valid ways to approach a block and the agent
    must commit to one. We preserve that by leaving the expert oracle's
    approach side stochastic.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        n_steps: int = 120,
        n_dim: int = 2,
        seed: Optional[int] = None,
        dt: float = 0.005,
        repeat_actions: int = 10,
        # PD gains chosen for ~critical damping (k_v = 2*sqrt(k_p) = 6.32).
        # Slightly over-damped (k_v=7) so the agent doesn't overshoot the
        # push-from waypoint and crash into the block from the wrong side.
        k_p: float = 10.0,
        k_v: float = 7.0,
        # Soft-contact spring constant: F_block = contact_k * overlap * dir.
        # Tuned so the block actually moves under the agent's PD control budget
        # without being so stiff that simulation explodes at dt=0.005.
        contact_k: float = 120.0,
        # Linear damping on the block. Without this the block coasts forever
        # after a single push. friction=2.5 gives an exponential half-life of
        # ~5 env-steps after release, which is short enough that the expert
        # can release ~0.15 units before the target and still land on it.
        block_friction: float = 2.5,
        agent_radius: float = 0.03,
        block_radius: float = 0.05,
        # IBC paper used ~5cm tolerance on a ~50cm workspace (10%). Match.
        goal_distance: float = 0.07,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_dim = n_dim
        self.dt = dt
        self.repeat_actions = repeat_actions
        self.k_p = k_p
        self.k_v = k_v
        self.contact_k = contact_k
        self.block_friction = block_friction
        self.agent_radius = agent_radius
        self.block_radius = block_radius
        self.contact_radius = agent_radius + block_radius
        self.goal_distance = goal_distance
        self.render_mode = render_mode

        self._rng = np.random.RandomState(seed=seed)

        # Action: agent position setpoint in [0,1]^n.
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_dim,), dtype=np.float32
        )
        self.observation_space = self._create_observation_space()

        self.steps = 0
        self.obs_log: list[dict] = []
        self.act_log: list[dict] = []
        self.min_dist_block_to_target = np.inf
        self.first_contact_step: Optional[int] = None

        self.fig = None
        self.ax = None

    def _create_observation_space(self) -> spaces.Box:
        # 6 fields of n_dim, see module docstring for layout.
        low = np.concatenate([
            np.zeros(self.n_dim),             # pos_agent
            np.full(self.n_dim, -np.inf),     # vel_agent
            np.zeros(self.n_dim),             # pos_block
            np.full(self.n_dim, -np.inf),     # vel_block
            np.zeros(self.n_dim),             # pos_target
            np.zeros(self.n_dim),             # pos_target dup
        ]).astype(np.float32)
        high = np.concatenate([
            np.ones(self.n_dim),
            np.full(self.n_dim, np.inf),
            np.ones(self.n_dim),
            np.full(self.n_dim, np.inf),
            np.ones(self.n_dim),
            np.ones(self.n_dim),
        ]).astype(np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(seed=seed)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.RandomState(seed=seed)

        self.steps = 0
        self.obs_log = []
        self.act_log = []
        self.min_dist_block_to_target = np.inf
        self.first_contact_step = None

        # Sample block and target far enough apart that the task is non-trivial.
        # The min separation keeps ~95% of the configuration space available
        # so the dataset still covers it.
        while True:
            pos_block = self._rng.uniform(0.15, 0.85, size=self.n_dim).astype(np.float32)
            pos_target = self._rng.uniform(0.15, 0.85, size=self.n_dim).astype(np.float32)
            if np.linalg.norm(pos_block - pos_target) > 0.25:
                break

        # Spawn the agent on the BACK side of the block (opposite the target),
        # within a wedge wide enough to still give the oracle a non-trivial
        # approach. Without this constraint a naive straight-line approach
        # to push_from cuts through the block — the simple oracle has no path
        # planning. Variance comes from the wedge: agent lands anywhere in a
        # ~120° arc behind the block. Captures the "many ways to approach"
        # multi-modality the IBC paper highlights without needing a planner.
        block_to_target = pos_target - pos_block
        btt_norm = float(np.linalg.norm(block_to_target))
        back_dir = -block_to_target / btt_norm
        while True:
            radial = self._rng.uniform(0.10, 0.30)
            # Random unit perturbation in the half-space behind the block.
            perturb = self._rng.randn(self.n_dim).astype(np.float32)
            perturb = perturb - perturb.dot(-back_dir) * (-back_dir)  # remove forward component
            perturb_norm = float(np.linalg.norm(perturb))
            if perturb_norm < 1e-6:
                continue
            perturb = perturb / perturb_norm
            tangent_mag = self._rng.uniform(-0.15, 0.15)
            offset = back_dir * radial + perturb * tangent_mag
            pos_agent = (pos_block + offset).astype(np.float32)
            if np.all(pos_agent > 0.02) and np.all(pos_agent < 0.98) \
               and np.linalg.norm(pos_agent - pos_block) > self.contact_radius + 0.02:
                break

        obs_dict = {
            "pos_agent": pos_agent,
            "vel_agent": np.zeros(self.n_dim, dtype=np.float32),
            "pos_block": pos_block,
            "vel_block": np.zeros(self.n_dim, dtype=np.float32),
            "pos_target": pos_target,
        }
        self.obs_log.append(obs_dict)
        return self._get_flat_observation(), {}

    def _get_flat_observation(self) -> np.ndarray:
        obs = self.obs_log[-1]
        return np.concatenate([
            obs["pos_agent"],
            obs["vel_agent"],
            obs["pos_block"],
            obs["vel_block"],
            obs["pos_target"],
            obs["pos_target"],  # duplicated, see module docstring
        ]).astype(np.float32)

    def _internal_step(self, action: np.ndarray):
        """Single dt integration: PD agent + soft-contact block + friction."""
        self.act_log.append({"pos_setpoint": action.copy()})
        obs = self.obs_log[-1]

        # Agent PD control toward setpoint.
        u_agent = self.k_p * (action - obs["pos_agent"]) - self.k_v * obs["vel_agent"]

        # Contact force on block (and reaction on agent, scaled down — the
        # paper's agent is much heavier than the block, so we damp the agent
        # reaction without making it exactly zero — keeps things physical).
        diff = obs["pos_block"] - obs["pos_agent"]
        dist = float(np.linalg.norm(diff))
        if dist < self.contact_radius and dist > 1e-8:
            direction = diff / dist
            overlap = self.contact_radius - dist
            contact_force = self.contact_k * overlap * direction
            f_on_block = contact_force
            f_on_agent = -0.1 * contact_force
            if self.first_contact_step is None:
                self.first_contact_step = self.steps
        else:
            f_on_block = np.zeros(self.n_dim, dtype=np.float32)
            f_on_agent = np.zeros(self.n_dim, dtype=np.float32)

        # Block dynamics: linear friction (Stokes drag) + contact force.
        u_block = f_on_block - self.block_friction * obs["vel_block"]

        new_pos_agent = obs["pos_agent"] + obs["vel_agent"] * self.dt
        new_vel_agent = obs["vel_agent"] + (u_agent + f_on_agent) * self.dt
        new_pos_block = obs["pos_block"] + obs["vel_block"] * self.dt
        new_vel_block = obs["vel_block"] + u_block * self.dt

        # Clamp positions to workspace [0, 1]^n. Reflect velocity component
        # at walls so a block pinned to a wall doesn't accumulate velocity
        # against the boundary.
        for arr_pos, arr_vel in (
            (new_pos_agent, new_vel_agent),
            (new_pos_block, new_vel_block),
        ):
            below = arr_pos < 0.0
            above = arr_pos > 1.0
            arr_vel[below] = np.maximum(arr_vel[below], 0.0)
            arr_vel[above] = np.minimum(arr_vel[above], 0.0)
        np.clip(new_pos_agent, 0.0, 1.0, out=new_pos_agent)
        np.clip(new_pos_block, 0.0, 1.0, out=new_pos_block)

        new_obs = copy.deepcopy(obs)
        new_obs["pos_agent"] = new_pos_agent.astype(np.float32)
        new_obs["vel_agent"] = new_vel_agent.astype(np.float32)
        new_obs["pos_block"] = new_pos_block.astype(np.float32)
        new_obs["vel_block"] = new_vel_block.astype(np.float32)
        self.obs_log.append(new_obs)

    def _dist_block_to_target(self) -> float:
        obs = self.obs_log[-1]
        return float(np.linalg.norm(obs["pos_block"] - obs["pos_target"]))

    @property
    def succeeded(self) -> bool:
        return self._dist_block_to_target() < self.goal_distance

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        action = np.clip(action, 0.0, 1.0).astype(np.float32)

        for _ in range(self.repeat_actions):
            self._internal_step(action)

        observation = self._get_flat_observation()
        terminated = self.steps >= self.n_steps
        self.min_dist_block_to_target = min(
            self.min_dist_block_to_target, self._dist_block_to_target()
        )
        # Sparse reward: 1.0 only if the block is at the target at the
        # FINAL step (consistent with the particle env's "stay at the goal").
        if terminated:
            reward = 1.0 if self.succeeded else 0.0
        else:
            reward = 0.0

        info = {
            "min_dist_block_to_target": self.min_dist_block_to_target,
            "final_dist_block_to_target": self._dist_block_to_target(),
            "success": self.succeeded if terminated else False,
            "first_contact_step": self.first_contact_step,
        }
        return observation, reward, terminated, False, info

    # ── Rendering ────────────────────────────────────────────────────────────
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        if self.n_dim == 2:
            return self._render_2d()
        return self._render_nd()

    def _render_2d(self) -> np.ndarray:
        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
            if self.render_mode == "human":
                plt.ion()
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect("equal")
        self.ax.set_title(f"Step {self.steps}/{self.n_steps}")

        target = self.obs_log[0]["pos_target"]
        self.ax.scatter(*target, c="green", s=250, marker="*", label="Target", zorder=5)
        self.ax.add_patch(plt.Circle(target, self.goal_distance, color="green", fill=False, ls="--"))

        block = self.obs_log[-1]["pos_block"]
        self.ax.add_patch(plt.Circle(block, self.block_radius, color="orange", fill=True, alpha=0.7))
        agent = self.obs_log[-1]["pos_agent"]
        self.ax.add_patch(plt.Circle(agent, self.agent_radius, color="red", fill=True, alpha=0.9))

        # Trajectory
        if len(self.obs_log) > 1:
            traj_a = np.array([o["pos_agent"] for o in self.obs_log])
            traj_b = np.array([o["pos_block"] for o in self.obs_log])
            self.ax.plot(traj_a[:, 0], traj_a[:, 1], "r-", alpha=0.4, lw=1, label="agent")
            self.ax.plot(traj_b[:, 0], traj_b[:, 1], color="orange", alpha=0.6, lw=1, label="block")

        self.ax.legend(loc="upper right")
        self.fig.canvas.draw()
        if self.render_mode == "human":
            self.fig.canvas.flush_events()
        buf = self.fig.canvas.buffer_rgba()
        return np.asarray(buf, dtype=np.uint8)[:, :, :3].copy()

    def _render_nd(self) -> np.ndarray:
        if self.fig is None:
            self.fig, self.ax = plt.subplots(self.n_dim, 1, figsize=(10, 2 * self.n_dim))
            if self.n_dim == 1:
                self.ax = [self.ax]
            if self.render_mode == "human":
                plt.ion()
        for d in range(self.n_dim):
            self.ax[d].clear()
            self.ax[d].set_xlim(0, self.n_steps)
            self.ax[d].set_ylim(0, 1)
            self.ax[d].set_ylabel(f"Dim {d}")
            if len(self.obs_log) > 1:
                ta = [o["pos_agent"][d] for o in self.obs_log]
                tb = [o["pos_block"][d] for o in self.obs_log]
                self.ax[d].plot(range(len(ta)), ta, "r-", label="agent")
                self.ax[d].plot(range(len(tb)), tb, color="orange", label="block")
            self.ax[d].axhline(y=self.obs_log[0]["pos_target"][d], color="green", ls="--", label="target")
        self.ax[-1].set_xlabel("Step")
        self.ax[0].legend()
        self.fig.canvas.draw()
        if self.render_mode == "human":
            self.fig.canvas.flush_events()
        buf = self.fig.canvas.buffer_rgba()
        return np.asarray(buf, dtype=np.uint8)[:, :, :3].copy()

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# ─── Expert oracle ───────────────────────────────────────────────────────────

def pushing_expert_action(
    pos_agent: np.ndarray,
    pos_block: np.ndarray,
    pos_target: np.ndarray,
    contact_radius: float,
    vel_block: Optional[np.ndarray] = None,
    approach_buffer: float = 0.04,
    align_tol: float = 0.12,
    stop_radius: float = 0.10,
) -> np.ndarray:
    """Heuristic oracle policy: navigate behind block, push toward target.

    The previous two-phase version oscillated because the agent's PD inertia
    sent it past push_from into the block from the wrong side. Fix: use the
    BLOCK's velocity to decide if we're already pushing correctly. Once the
    block is moving toward the target, commit to push-through mode regardless
    of fine-grained alignment.

    Modes:
      * "Approach" — agent isn't near the block, OR block velocity component
        toward the target is small/negative: drive setpoint to push_from.
      * "Push" — agent is near and behind the block AND block velocity has a
        positive component toward target: drive setpoint to a point past the
        target so contact stays as the block moves.

    `vel_block` is optional: if not provided we fall back to a geometric-only
    test (used during single-step planning where vel is unknown).

    Returns: action in [0, 1]^n.
    """
    block_to_target = pos_target - pos_block
    btt_norm = float(np.linalg.norm(block_to_target))
    if btt_norm < 1e-6:
        return pos_block.copy()
    dir_bt = block_to_target / btt_norm

    # Stop condition: once the block is within stop_radius, retreat. Residual
    # block momentum dissipates via friction rather than carrying the block
    # past the target. We retreat to a point behind the block AND in the
    # SAME perpendicular plane — so the agent doesn't kick the block sideways
    # while disengaging.
    if btt_norm < stop_radius:
        retreat = pos_block - 2.5 * contact_radius * dir_bt
        return retreat.astype(np.float32)

    push_from = pos_block - (contact_radius + approach_buffer) * dir_bt

    agent_to_block = pos_block - pos_agent
    along = float(np.dot(agent_to_block, dir_bt))
    perp = agent_to_block - along * dir_bt
    perp_dist = float(np.linalg.norm(perp))
    agent_block_dist = float(np.linalg.norm(agent_to_block))

    block_moving_right_way = False
    if vel_block is not None:
        block_v_along = float(np.dot(vel_block, dir_bt))
        block_moving_right_way = block_v_along > 0.05

    in_push_zone = (
        agent_block_dist < contact_radius + approach_buffer + 0.02
        and along > 0
        and perp_dist < align_tol
    )

    if in_push_zone or block_moving_right_way:
        # Pressure-setpoint: aim PAST the block in the push direction with a
        # forward_offset that scales down as we approach the target so residual
        # momentum after release doesn't carry the block past. Linear ramp
        # from 0.10 at btt=0.40 to 0.0 at btt=stop_radius.
        ramp_start = 0.40
        forward_offset = float(np.clip(
            0.10 * (btt_norm - stop_radius) / (ramp_start - stop_radius),
            0.0, 0.10,
        ))
        return (pos_block + forward_offset * dir_bt).astype(np.float32)
    return push_from.astype(np.float32)


# ─── Registration ────────────────────────────────────────────────────────────

gym.register(
    id="Pushing-v0",
    entry_point="simulations.pushing_env:PushingEnv",
    max_episode_steps=120,
)


if __name__ == "__main__":
    # Quick sanity check: oracle should drive block to target.
    print("Smoke-testing PushingEnv + expert oracle ...")
    env = PushingEnv(n_dim=2, render_mode=None)
    successes = 0
    n_test = 30
    for seed in range(n_test):
        obs, _ = env.reset(seed=seed)
        for _ in range(env.n_steps):
            o = env.obs_log[-1]
            act = pushing_expert_action(
                o["pos_agent"],
                o["pos_block"],
                o["pos_target"],
                env.contact_radius,
                vel_block=o["vel_block"],
            )
            obs, r, term, trunc, info = env.step(np.clip(act, 0, 1))
            if term or trunc:
                break
        successes += int(info.get("success", False))
    print(f"Expert success rate: {successes}/{n_test} = {100 * successes / n_test:.0f}%")
    env.close()
