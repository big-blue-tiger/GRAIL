from types import SimpleNamespace

from omegaconf import OmegaConf
import torch

from gear_sonic.eval_agent_trl import (
    _apply_run_once_motion_cap,
    _evaluation_motion_keys,
    _evaluation_reset_reasons,
    _evaluation_step_dt,
)


def test_run_once_caps_motion_library_before_environment_creation():
    config = OmegaConf.create(
        {
            "run_once": True,
            "load_only_num_envs_motions": True,
            "num_envs": 8,
            "manager_env": {
                "commands": {
                    "motion": {"motion_lib_cfg": {"max_unique_motions": None}}
                }
            },
        }
    )

    assert _apply_run_once_motion_cap(config) == 8
    assert config.manager_env.commands.motion.motion_lib_cfg.max_unique_motions == 8

    # Preserve a more restrictive explicit user cap.
    config.manager_env.commands.motion.motion_lib_cfg.max_unique_motions = 4
    assert _apply_run_once_motion_cap(config) == 4


def test_eval_reset_reporting_uses_motion_key_term_and_environment_dt():
    class TerminationManager:
        active_terms = ["object_pos_deviation", "motion_time_out"]

        @staticmethod
        def get_term(name):
            return {
                "object_pos_deviation": torch.tensor([True, False]),
                "motion_time_out": torch.tensor([False, True]),
            }[name]

    command = SimpleNamespace(
        motion_ids=torch.tensor([1, 0]),
        motion_lib=SimpleNamespace(curr_motion_keys=["sample_a", "sample_b"]),
    )
    env = SimpleNamespace(
        motion_command=command,
        env=SimpleNamespace(
            termination_manager=TerminationManager(),
            step_dt=0.02,
        ),
    )

    assert _evaluation_motion_keys(env, 2) == ["sample_b", "sample_a"]
    assert _evaluation_reset_reasons(env, 0) == ["object_pos_deviation"]
    assert _evaluation_reset_reasons(env, 1) == ["motion_time_out"]
    assert _evaluation_step_dt(env) == 0.02
