# Reference-free joint_micro_step student play

Run from `imports/SONIC` with the Isaac Lab Python environment. The checkpoint
must use `robocasa_pickup_table_mlp_decoder_latent_vector_obs_joint_micro_step`.

```bash
python gear_sonic/eval_agent_trl.py \
  +checkpoint=/home/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/0308Termination_walk-20260920_115339/model_step_005400.pt \
  +headless=True ++num_envs=32 +run_once=True \
  reference_free.enabled=True \
  reference_free.dataset_dir=/home/GRAIL/data/hf_dataset/data_update/data/pickup_table_walk_concat \
  reference_free.initial_pose_file=/home/GRAIL/imports/SONIC/models/init_g1_state/pickup_table__alcohol_1__000_first_frame.pkl \
  ++manager_env.config.render_results=True \
  ++manager_env.config.save_rendering_dir=/home/GRAIL/outputs/0308Termination_walk-reference-free \
  '~manager_env/recorders=empty' '+manager_env/recorders=render'
```

No `motion_file`, `object_motion_file`, motion-library asset path or tracking
termination override is needed. Only the selected object files are briefly read
to extract the initial height and orientation; no motion library is constructed.
Objects, BPS and table metadata are paired by filename. Each environment keeps
its assigned object while reset samples a new robot position, object XY and goal.

The goal keeps the object's initial XY and orientation and adds 0.4–0.7 m to its
initial Z. It is supplied through the existing 18-dimensional target-pose term
inside the student's 66-dimensional privileged observation. Body/hand state and
decoder histories retain their training order. Initial robot roll/pitch is kept,
yaw faces the table, and root height includes the configured 0.05 m compensation.

Quaternion conventions follow training: robot `root_rot` is converted from
`xyzw` to Isaac Lab `wxyz`; object `root_quat` and metadata `table_quat` are
passed through as in `TrackingCommand`. Robot local +X is forward, +Y is left,
and +Z is up. Sampling offsets use world axes. Relative target positions use
`R_frame.T @ (target_world - frame_world)`; rotations use the same inverse frame
rotation, with pelvis then hand and row-major first-two-column 6D encoding.

Blue marks the robot start line, orange marks the object sampling rectangle,
and green spheres mark goals. `reference_free.debug_vis=False` hides these markers.
`reference_free.success_threshold=0.1` is the three-dimensional goal tolerance.
`++manager_env.config.episode_length_s=10` controls the time limit.

Each environment ends its episode on reaching the goal or timing out. With
`reference_free.terminate_on_success=False`, reaching the goal does not reset
the environment; it continues until the time limit. Success is evaluated from
the final object position, so reaching the goal earlier does not guarantee success.
With
`run_once=True`, only its first episode counts. Videos use the existing render
recorder; `reference_free_results.json` is written to the rendering directory
(override with `+run_once_report_path=...`). Positions in the report are world
coordinates, including environment origins. A timeout can be successful if
the final object position is within the goal tolerance.

CPU regression checks:

```bash
python -m unittest discover -s tests -p test_reference_free.py
python tests/check_reference_free_runtime.py
```
