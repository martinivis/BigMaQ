# Action Recognition

## Rerunning the Experiment with Modified Pose Data

If you have re-optimized the poses for the entire dataset, you can rerun the action recognition pipeline using the updated pose information.

1. Copy the updated parameter files using `copy_dataset.py`.
2. Extend the pose representations to rotation matrices with `extend_pose_info.py`.
3. Optionally, copy the required activation files from external storage to your fast-access storage to improve processing speed.
4. If storage space becomes limited, use `delete_some_activations.py` to remove unnecessary activation files and free up disk space.
5. Rerun the action recognition benchmarks using:

   * `benchmarking.py`
This workflow allows you to efficiently evaluate the impact of modified pose data while managing storage constraints on the fast-access drive.
