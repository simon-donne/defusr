# Learning Non-volumetric Depth Fusion using Successive Reprojections

This is the github page for the paper

    Learning Non-volumetric Depth Fusion using Successive Reprojections
    Simon Donné, Andreas Geiger
    CVPR 2019

All relevant code is available here.
For the datasets, videos, and so on, please visit https://avg.is.tuebingen.mpg.de/research_projects/defusr.

## Requirements / Setting up

### Creating a working conda environment

Unfortunately, PyTorch changed their C++ bindings halfway through this project.
The current version of this code only supports PyTorch 0.4.1.

    conda create -n defusr python=3.7
    conda activate defusr
    conda install pytorch=0.4.1 torchvision cuda92 -c pytorch
    conda install -c conda-forge opencv
    pip install pycuda
    pip install Cython
    conda install termcolor
    conda install matplotlib
    conda install gitpython
    conda install tqdm

### Compiling the MYTH library

MYTH (My Torch Helpers) is an included library of auxiliary functions required for the network execution.
It can be compiled by

    conda activate defusr
    cd code/MYTH
    chmod +x build.sh
    ./build.sh


## Running the evaluation

For running the evaluation, download the pretrained networks [here](https://avg.is.tuebingen.mpg.de/research_projects/defusr), as well as the DTU dataset.
Fix the relevant paths in the scripts in the `local_config.py` file, and run the scripts.
First, however, the colmap and MVSNet outputs will need to be created. Please refer to the relevant scripts in the data/ folder for this.
For the COLMAP fusion, a patch to the base COLMAP version is necessary (which allows fusion without sparse matches, i.e. from the manually generated fusion lists).
You can find the patch [here](data/colmap_patch.patch).

## External dependencies

For the MVSNet repository, I suggest checking out commit 9284c4bc8. The scripts provided here interface with the output created by that commit.
For the COLMAP repository, I suggest checking out commit c13602fe5. Similarly, the patch provided above also acts correctly on that commit.
