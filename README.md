# SARD
This repo contains the official implementation of our paper:

**Symmetry-Aware Robot Design with Structured Subgroups**. (Submitted to ICML 2023)


# Installation 

### Environment
* Tested OS: Linux
* Python >= 3.7
* PyTorch == 1.8.0
### Dependencies:
1. Install [PyTorch 1.8.0](https://pytorch.org/get-started/previous-versions/) with the correct CUDA version.
2. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Install torch-geometric with correct CUDA and PyTorch versions (change the `CUDA` and `TORCH` variables below): 
    ```
    CUDA=cu102
    TORCH=1.8.0
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-geometric==1.6.1
    ```
4. install mujoco-py following the instruction [here](https://github.com/openai/mujoco-py#install-mujoco).


# Training
You can train your own models using the provided config in [design_opt/cfg](design_opt/cfg):
```
python design_opt/train.py 
--cfg derl
--derl_cfg design_opt/cfg/derl_configs/eval/point_nav.yml
--num_threads 20
--enable_wandb 1
--seed 0
--enable_infer 1
```
You can replace `point_nav.yml` with {`escape.yml`, `patrol.yml`, `locomotion_vt.yml`, `locomotion_ft.yml`, `manipulation_box.yml`} to train other environments. Here is the correspondence between the configs and the environments in the paper: `Point Navigation`, `Escape Bowl`, `Patrol`, `Locomotion on Variable Terrain`, `Locomotion on Flat Terrain` and `Manipulate Box`.

Our experiment is logged online with `wandb`, and set `--enable_wandb` to `0` if you want to disable it.

# Visualization
If you have a display, run the following command to visualize the pretrained model for the `Point Navigation`:
```
python design_opt/eval.py 
--cfg derl
--derl_cfg design_opt/cfg/derl_configs/eval/point_nav.yml
--expID 2023_01_06-07:49:44
--epoch best
--enable_infer 1
--pause_design
```
Again, you can replace `Point Navigation` with {`escape.yml`, `patrol.yml`, `locomotion_vt.yml`, `locomotion_ft.yml`, `manipulation_box.yml`} to visualize other environments.

# License
Please see the [license](LICENSE) for further details.
