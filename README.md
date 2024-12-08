Table of Contents
=================

   * [Overview](#iwisdm)
   * [Examples](#examples)
   * [Usage](#usage)
      * [Install Instructions](#install-instructions)
      * [Environment Init](#shapenet-environment-initialization)
      * [Basic Usage](#basic-usage)
   * [Acknowledgements](#acknowledgements)
   * [Citation](#citation)
   
# iWISDM
iWISDM, short for instructed-Virtual VISual Decision Making, is a virtual environment capable of generating a limitless array of _vision-language tasks with varying complexity_. 

It is a toolkit designed to evaluate the ability of multimodal models to follow instructions in visual tasks. It builds on the compositional nature of human behavior, and the fact that complex tasks are often constructed by combining smaller task units together in time.

iWISDM encompasses a broad spectrum of subtasks that engage executive functions such as inhibition of action, working memory, attentional set, task switching, and schema generalization. 

It is also a scalable and extensible framework which allows users to easily define their own task space and stimuli dataset.

# Examples
Using iWISDM, we have compiled four distinct benchmarks of increasing complexity levels for evaluation of large multi-modal models.

Below is an example of the generated tasks:
<p align="center">
  <img width="800" alt="Screenshot 2024-06-24 at 8 43 41 PM" src="https://github.com/BashivanLab/iWISDM/assets/44264329/5f7eeffe-a3be-405f-8514-6424818cf5b7">
</p>
<p align="center">
  <img src="https://github.com/BashivanLab/iWISDM/blob/main/benchmarking/param_table.png?raw=true" alt="benchmarking params"/>
</p>

These datasets can be generated from [/benchmarking](https://github.com/BashivanLab/iWISDM/tree/main/benchmarking) or downloaded at: 
[iWISDM_benchsets.tar.gz](https://drive.google.com/file/d/1K-9AAJfvz6kiN3h9X2Rg0D88gJQ_rxSu/view?usp=sharing)

iWISDM inherits several classes from COG ([github.com/google/cog](https://github.com/google/cog)) to build task graphs. For convenience, we have also pre-implemented several commonly used cognitive tasks in task_bank.py. 


### For further details, please refer to our paper at:
[https://arxiv.org/submit/5678755/view](https://arxiv.org/abs/2406.14343)

# Usage
### Install Instructions
To install the iWISDM package, simply run the following command:
```shell
pip install iwisdm
```
If you would like to install the package from source, you can clone the repository and follow the instructions below:
#### Install Poetry
```shell
curl -sSL https://install.python-poetry.org | python3 -
```
#### Create conda python environment
```shell
conda create --name iwisdm python=3.11
```
#### Install dependencies
```shell
poetry install
```

### ShapeNet Environment Initialization
To initialize the ShapeNet environment, you will need to download the ShapeNet dataset, this is for rendering the trials.

To replicate our experiments, you also need to download the benchmarking configurations.

ShapeNet is a large-scale repository of shapes represented by 3D CAD models of objects  [(Chang et. al. 2015)](https://arxiv.org/abs/1512.03012).
#### Pre-rendered Dataset Download
[shapenet_handpicked.tar.gz](https://drive.google.com/file/d/1is72QDjP6A6TA1mZLL3doYWaU08waAxm/view?usp=sharing) 

#### Benchmarking Configs Download
[configs.tar.gz](https://github.com/BashivanLab/iWISDM/tree/main/benchmarking/configs.tar.gz)
### Basic Usage

```python
# imports
from iwisdm import make
from iwisdm import read_write

# environment initialization
with open('your/path/to/env_config', 'r') as f:
    config = json.load(f)  # using pre-defined AutoTask configuration
env = make(
    env_id='ShapeNet',
    dataset_fp='your/path/to/shapenet_handpicked',
)
env.set_env_spec(
    env.init_env_spec(
        auto_gen_config=config,
    )
)

# AutoTask procedural task generation and saving trial
tasks = env.generate_tasks(10)  # generate 10 random task graphs and tasks
_, (_, temporal_task) = tasks[0]
trials = env.generate_trials(tasks=[temporal_task])  # generate a trial
imgs, _, info_dict = trials[0]
read_write.write_trial(imgs, info_dict, f'output/trial_{i}')
```

#### See [/tutorials](https://github.com/BashivanLab/iWISDM/tree/main/tutorials) for more examples.

# Acknowledgements
This repository builds upon the foundational work presented in the COG paper [(Yang et al.)](https://arxiv.org/abs/1803.06092).

Yang, Guangyu Robert, et al. "A dataset and architecture for visual reasoning with a working memory." Proceedings of the European Conference on Computer Vision (ECCV). 2018.

# Citation
If you find iWISDM useful in your research, please use the following BibTex:
```
@inproceedings{lei2024iwisdm,
  title={iWISDM: Assessing instruction following in multimodal models at scale},
  author={Lei, Xiaoxuan and Gomez, Lucas and Bai, Hao Yuan and Bashivan, Pouya},
  booktitle={Conference on Lifelong Learning Agents (CoLLAs 2024)},
  year={2024}
}
```
