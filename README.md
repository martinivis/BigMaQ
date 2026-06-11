
# BigMaQ


<p align="center">
  <img src="assets/output.gif" width="700">
</p>

<p>
<p align="center">
<a href="https://arxiv.org/abs/2602.19874">📄 Paper</a> |
<a href="https://docs.google.com/forms/d/e/1FAIpQLSf30FV5pyhGQac8C5tSM2dW17n7j4xpB_2sNu9UnkeBYdW93Q/viewform?usp=sharing&ouid=116164636450166882978">📊 Dataset request </a> |
<a href="https://martinivis.github.io/BigMaQ/">🌐 Project Page</a>
</p>

## Introduction

BigMaQ is a large-scale dataset for 3D macaque motion capture, pose estimation, and behavioral understanding. The dataset comprises more than 750 multi-view recordings of interacting rhesus macaques captured with 16 calibrated cameras, together with detailed 3D skeletal motion annotations, subject-specific textured avatars, and curated behavioral labels.

By integrating surface-based pose and shape representations into animal action recognition, BigMaQ moves beyond sparse keypoint descriptions and enables a richer characterization of posture, motion, and social interaction. The dataset also introduces BigMaQ500, a benchmark linking image observations with 3D pose representations for cross-subject action recognition.

BigMaQ is intended as a resource for research in computer vision, graphics, neuroscience, ethology, and animal behavior analysis, supporting the development of robust methods for markerless motion capture and behavioral recognition in non-human primates.

## Installation

### Environment

The code has been tested on Ubuntu 20.04 with CUDA 11.6, Python 3.9, PyTorch 1.13.0 and PyTorch3D 0.7.2.
Please create a conda environment for your specific setup according to [Pytorch3D installation](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
Afterwards, you can install the remaining packages for the environment listed in `pose_reconstruction/requirements.txt`, via:

```shell
pip install -r requirements.txt
```

If you want to optimize the pose in a multi-view consistent fashion similar to this paper [macaquepose3d](https://www.science.org/doi/10.1126/sciadv.adn1355) ,
you should also install the pictorial package:

```shell
cd /BigMaQ/pose_reconstruction/src/m_lib
python -m pip install cython numpy
python setup.py build_ext --inplace
python -m pip install -e .
```

## Code


The repository is organized as shown below. Before running either the action recognition or pose reconstruction pipelines, please configure the dataset location in the corresponding JSON configuration files within the `cfgs` folders.
To access the full dataset, please submit a data request [here](https://docs.google.com/forms/d/e/1FAIpQLSf30FV5pyhGQac8C5tSM2dW17n7j4xpB_2sNu9UnkeBYdW93Q/viewform?usp=sharing&ouid=116164636450166882978).
If you are only interested in the reconstructed pose priors and mesh models, please refer to the **Pose Priors Only** section below.

```text
.
├── pose_reconstruction
│   ├── cfgs/      # Configuration files
│   ├── data/      # Dataset and data-related resources
│   ├── scripts/   # Utility and processing scripts
│   └── src/       # Core source code
│
└── action_recognition
    ├── cfgs/      # Configuration files
    ├── model/     # Model architectures
    ├── scripts/   # Training, evaluation, and data processing scripts
    └── src/       # Core source code and utilities
```

### Dataset Structure

The dataset is divided into the original BigMaQ recordings (`BigMaQ_rec`) and the BigMaQ500 action recognition benchmark (`BigMaQ500`).

```text
<BigMaQ>
├── BigMaQ_rec
│   ├── dataset_overview.csv
│   ├── Calibration/
│   ├── ColorCalibration/
│   ├── IndividualFits/
│   ├── Session1/
│   ├── ...
│   └── BigMaQ_latest_pose_reconstructions/
│
└── BigMaQ500
    ├── Actions/
    └── tracked_actions.csv
```

The BigMaQ recordings are organized into recording sessions. Each session corresponds to entries in `dataset_overview.csv`. Geometric camera calibrations are stored in `Calibration/`, color calibration files in `ColorCalibration/`, and low- and high-poly individual fits in `IndividualFits/`.

The latest reconstructed poses are provided in `BigMaQ_latest_pose_reconstructions/`. Each action is stored in a separate folder containing the reconstructed pose parameters used for visualization, stimulus generation, and action recognition experiments.

The low- and high-poly base mesh models are included in the code project under:

```text
pose_reconstruction/data/Mesh/
```

### Pose Priors Only

If you are only interested in the reconstructed pose priors and do not require access to the original video recordings, we provide a lightweight archive containing the latest pose reconstructions of the BigMaQ dataset.

Compared to the reconstructions used in the paper, this release additionally incorporates joint angle limits and multi-view keypoint tracklet processing.

To match the expected directory structure, create the following folders and extract the archive into `BigMaQ_rec/`:

```text
BigMaQ/
├── BigMaQ_rec/
│   └── BigMaQ_latest_pose_reconstructions/
|   dataset_overview.csv
```

The latest pose reconstructions can be downloaded from:

https://zenodo.org/records/20649325


### Pose Reconstruction
- Before running any scripts with video data, please update the following configuration file, `pose_reconstruction/cfgs/Setup_Local_cfg.json`. Set the value of `hard_drive_loc` to the location of your dataset on disk.
- To run the script, please activate the environment, navigate into the folder `pose_reconstruction/scripts` and run the following in the command line
```code
python ActionTracking.py
```
- You can further change the number of cameras in optimization by `--nb_cameras 10`, here exemplary shown for 10 cameras
- To optimize only a specific action instead of iterating through the entire dataset, use `--specific-action-index 0`, which, in this case, will optimize/load the first action.
- By setting the flag `--high-poly`, you can render the high-poly mesh, but increases optimization time
- By additionally setting the flag `--render-col`, you can render the mesh in color.
- Rendered images are found for the specific action in the folder `Optimization_Renderings`. After optimization the same folder contains videos of the surface tracks from two views.

In case you want to optimize the pose reconstruction yourself, please consider additional information
[here](pose_reconstruction/README.md).

### Stimulus Rendering

You can use the reconstructed pose priors to render arbitrary actions from arbitrary viewpoints and with arbitrary macaque identities. An example is provided in `pose_reconstruction/scripts/ActionRendering.py`.

Run the script from within `pose_reconstruction/scripts`:

```bash
python ActionRendering.py
```

The arguments `--specific-action-index`, `--high-poly`, and `--render-col` behave identically to the pose reconstruction pipeline.

Additional rendering-specific arguments include:

- `--individual-index 0` selects which individual from a multi-animal action to render.
- `--display-individual L` renders the selected action using the body shape and appearance of another individual (`J`, `H`, `L`, `O`, `N`, `T`, `C`, `G`).
- `--frame-idx 50` renders a specific frame of the reconstructed action.
- `--azim`, `--elev`, and `--dist` control the camera viewpoint.

Rendered images are saved to the corresponding action folder in the pose reconstruction directory.


### Action Recognition

In the dataset BigMaQ500, we provide for each Action the original pose rotation angles, 2D keypoints, 3D keypoints, and mesh vertices. Additionally, we provide only precomputed `vit-base-cls` tokens to keep the dataset size manageable.

#### Training

- To run the benchmark, activate the environment, navigate to `action_recognition/scripts`, and execute:

```bash
python benchmarking.py
```

The input modality can be selected using `--run-idx`:

- `--run-idx 0`: Visual-only model (ViT features)
- `--run-idx 1`: Pose-only model
- `--run-idx 2`: Visual + Pose model

#### Evaluation

To aggregate the trained model results and report mean Average Precision (mAP), run:

```bash
python compute_map_per_category.py
```

The evaluation script loads the stored `metrics.csv` files and reports overall and category-wise mAP values.

##### Arguments

- `--run-idx`
  - `0`: Visual-only
  - `1`: Pose-only
  - `2`: Visual + Pose

- `--pose-idx`

  Selects the pose representation to evaluate:

  - `0`: `3D-AA` (joint rotation angles)
  - `1`: `3D-KP` (3D keypoints)
  - `2`: `3D-Vert` (mesh vertices)
  - `3`: `2D-KP` (2D keypoints)

Example:

```bash
python compute_map_per_category.py --run-idx 1 --pose-idx 2
```

This evaluates the pose-only benchmark using the reconstructed mesh vertices (`3D-Vert`).

For pose-only experiments with 3D pose representations (`3D-AA`, `3D-KP`, `3D-Vert`), results are averaged over 5 folds. All other configurations are averaged over 3 random seeds.

**Note**: The Python environment used for action recognition was based on newer PyTorch/CUDA versions than those typically supported by PyTorch3D and used for pose reconstruction. If you would like to reproduce the exact results reported in the paper, please refer to the dependency versions listed in `action_recognition/requirements.txt` and consider using a separate environment for action recognition.



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
If you find this dataset and code useful for your research, or use data generated by our model, 
please cite as follows:
```code
@inproceedings{
martini2026bigmaq,
title={BigMaQ: A Big Macaque Motion and Animation Dataset Bridging Image and 3D Pose Representations},
author={Lucas Martini and Alexander Lappe and Anna Bogn{\'a}r and Rufin Vogels and Martin A. Giese},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=n7viYE7Xbo}
}
```
## References

This project builds upon several open-source research efforts in 3D animal pose estimation and reconstruction. In particular, portions of the implementation, design choices, and processing pipeline were adapted from or inspired by the following works:

- [3D Bird Reconstruction: A Dataset, Model, and Shape Recovery from a Single View](https://github.com/marcbadger/avian-mesh)

- [Anipose: A Toolkit for Robust Markerless 3D Pose Estimation](https://github.com/lambdaloop/anipose) 

- [Three-dimensional markerless motion capture of multiple freely behaving monkeys toward automated characterization of social behavior](https://github.com/PrimatoModelling/macaque3Dpose/tree/main)

