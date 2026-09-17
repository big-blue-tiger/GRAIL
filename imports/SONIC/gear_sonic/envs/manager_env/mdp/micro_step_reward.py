"""Contact-based penalty for short, brief foot relocations."""

import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg


class MicroStepPenalty(ManagerTermBase):
    """Charge once at touchdown, only after observing a stance and a takeoff.

    Takeoff XY is the last observed stance position; touchdown XY is the
    current foot position. Air duration comes from the contact sensor, not
    foot height. The sensor must enable ``track_air_time``. Contact transitions
    are observed at policy frequency, so sub-control-step swings are ignored.

    Returns the number of qualifying foot touchdowns in this control step.
    RewardManager multiplies by weight and step_dt, so each qualifying foot
    contributes weight * step_dt to the reward at touchdown.
    Reference-speed gating is disabled; short steps are evaluated at any speed.
    It does not infer whether an adjustment is necessary for the task.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        params = cfg.params
        if params["min_step_length"] <= 0 or params["min_air_time"] <= 0:
            raise ValueError("Micro-step length and air-time thresholds must be positive")
        # Reference-speed gating is disabled.
        # if not 0 <= params["slow_speed_full"] < params["slow_speed_zero"]:
        #     raise ValueError("Expected 0 <= slow_speed_full < slow_speed_zero")
        sensor_cfg = params["sensor_cfg"]
        asset_cfg = params["asset_cfg"]
        sensor = env.scene.sensors[sensor_cfg.name]
        asset = env.scene[asset_cfg.name]
        if not sensor.cfg.track_air_time:
            raise ValueError("MicroStepPenalty requires contact sensor track_air_time=True")
        sensor_names = [sensor.body_names[i] for i in sensor_cfg.body_ids]
        asset_names = [asset.body_names[i] for i in asset_cfg.body_ids]
        if len(sensor_names) != 2 or sensor_names != asset_names:
            raise ValueError("Select the same two ordered foot bodies in sensor_cfg and asset_cfg")
        shape = (env.num_envs, 2)
        self._initialized = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self._was_contact = torch.zeros_like(self._initialized)
        self._valid_swing = torch.zeros_like(self._initialized)
        self._previous_xy = torch.zeros((*shape, 2), device=env.device)
        self._takeoff_xy = torch.zeros_like(self._previous_xy)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._initialized[ids] = False
        self._was_contact[ids] = False
        self._valid_swing[ids] = False
        self._previous_xy[ids] = 0
        self._takeoff_xy[ids] = 0

    def __call__(
        self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
        min_step_length: float, min_air_time: float,
        # Accept legacy configs, but these gate parameters are currently ignored.
        command_name: str = "motion_velocity", slow_speed_full: float = 0.05,
        slow_speed_zero: float = 0.10, use_slow_speed_gate: bool = False,
    ):
        sensor = env.scene.sensors[sensor_cfg.name]
        xy = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, :2]
        contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
        takeoff = self._initialized & self._was_contact & ~contact
        touchdown = self._initialized & ~self._was_contact & contact & self._valid_swing
        self._takeoff_xy.copy_(torch.where(takeoff[..., None], self._previous_xy, self._takeoff_xy))
        length = torch.linalg.vector_norm(xy - self._takeoff_xy, dim=-1)
        air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
        cost = (
            touchdown * (length < min_step_length)
            * (air_time < min_air_time)
        ).sum(dim=-1)
        self._valid_swing.copy_((self._valid_swing | takeoff) & ~contact)
        self._was_contact.copy_(contact)
        self._previous_xy.copy_(xy)
        self._initialized.fill_(True)
        return cost
