from transformers import TrainerCallback


class SegmentStopCallback(TrainerCallback):
    """Stop at an absolute iteration while retaining the full training horizon."""

    def __init__(self, stop_iteration):
        self.stop_iteration = int(stop_iteration)
        if self.stop_iteration <= 0:
            raise ValueError("stop_iteration must be positive")

    def on_train_begin(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_iteration:
            raise ValueError(
                f"Checkpoint iteration {state.global_step} already reached "
                f"segment target {self.stop_iteration}"
            )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        # The script configures ModelSaveCallback to save both checkpoints here.
        if state.global_step >= self.stop_iteration:
            control.should_training_stop = True
        return control
