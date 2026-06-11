# Pose Reconstruction

## Configuration File

The file

```text
pose_reconstruction/cfgs/Setup_Local_cfg.json
```

contains several settings controlling data loading, debugging, rendering, and output generation:

```json
{
    "force_action_reload": false,
    "disable_saving": false,
    "epoch_render_mod": 400,
    "render_video": true,
    "hard_drive_loc": "/path/to/dataset"
}
```

### `force_action_reload`

When set to `true`, the action is reloaded from the original dataset annotations and labels, even if previously optimized pose parameters are already available on disk.

This is useful when:

* Labels or annotations have changed.
* Cached action data should be ignored.
* A complete re-processing of the action is desired, which will depending on settings overwrite the existing parameters!

### `disable_saving`

Controls whether loaded action labels are written to disk (increases disk storage!).

* `false`: loaded labels can be cached and saved for faster future access.
* `true`: disables saving of loaded labels and action data.

This option is mainly useful for debugging or testing changes without modifying cached files.

### `epoch_render_mod`

Determines how often intermediate renderings are generated during optimization.

For example:

```json
{
    "epoch_render_mod": 100
}
```

renders debugging visualizations every 400 optimization epochs.

Smaller values provide more frequent visual feedback but increase runtime and disk usage.

### `render_video`

Controls whether videos are generated after optimization.

* `true`: creates visualization videos showing the reconstructed surface tracks after optimization.
* `false`: skips video generation.


### `hard_drive_loc`

Path to the dataset root directory.

Example:

```json
{
    "hard_drive_loc": "/media/lucas/FastInternal/BigMaQ/BigMaQ_rec"
}
```

The pose reconstruction pipeline expects the dataset and associated files to be located within this directory.
