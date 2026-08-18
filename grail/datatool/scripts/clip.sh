cd /home/tide/robot/GRAIL
conda activate grail

python -u -m grail.visualization.prepare_vis_shard \
    --data_dir data/hf_dataset/data/pickup_table_update \
    --shard_dir /tmp/pickup_table_update_vis_shard \
    --quat_convention xyzw

python -u grail/datatool/batch_render_replay_clip.py \
    --shard_dir /tmp/pickup_table_update_vis_shard \
    --traj_dir /tmp/pickup_table_update_vis_shard/trajectories \
    --object_usd_dir data/hf_dataset/data/pickup_table_update/object_usd \
    --output_dir data/hf_dataset/data/pickup_table_update/vis_penetration_check \
    --resolution 1920x1080 \
    --start_frame_skip 0 \
    --headless \
    --segmented_output_dir data/hf_dataset/data/pickup_table_update/clip_robot \
    --segmented_object_output_dir data/hf_dataset/data/pickup_table_update/clip_object \
    --no_record_video
