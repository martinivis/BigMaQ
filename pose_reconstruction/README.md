# BigMac3D: The Big Macaque 3D Motion and Animation Dataset

This project showcases the fitting process of pose and shape for macaques over time.


## Installation

### Environment

The code has been tested on Ubuntu 20.04 with CUDA 11.6, Python 3.9, PyTorch 1.13.0 and PyTorch3D 0.7.2.
Please create a conda environment for your specific setup according to [Pytorch3D installation](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
Afterwards, you can install the remaining packages for the environment listed in requirements.txt, via:

```shell
    pip install -r requirements.txt
```

## Folder Structure
```
BigMac3D
│   README.md
│───cfgs
│───data
│   └───Action 
│   └─── ...
│───scripts
│   └───ActionTracking.py   
│───src
│   └───Optimizers
│   └───utils
```

## Code Execution

- Please download first tracking data from [here](https://www.dropbox.com/scl/fo/ogg7bmm1q8czwj7so5pp9/ADlqKAi_WengbY09RSR-SCk?rlkey=0r6uq7za4jszze0ynm8pge48q&st=12740y7n&dl=0), and unzip it in the data folder.
Then the _Action_ folder should exist as shown in the folder structure above.
- To run the script, please activate the environment, navigate into the folder _scripts_ of BigMac3D and run the following in the command line
```code
python ActionTracking.py
```
- You can further change the number of cameras in optimization by `--nb_cameras 10`, here exemplary shown for 10 cameras
- By setting the flag `--high-poly`, you can render the high-poly mesh, but increases optimization time
- By additionally setting the flag `--render-col`, you can render the mesh in color.
- Rendered images are found in `data/Action/Optimization_Renderings`. After optimization the same folder contains vidos of the surface tracks from two views.



