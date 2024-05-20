# iWISDM
iWISDM, short for instructed-Virtual VISual Decision Making, is a virtual environment capable of generating a limitless array of _vision-language tasks with varying complexity_. iWISDM encompasses a broad spectrum of tasks that engage executive functions – such as inhibition of action, working memory, attentional set, task switching, and schema generalization. It is also a scalable and extensible framework which allows users to easily define their own task space and stimuli dataset. iWISDM builds on the compositional nature of human behavior, and the fact that complex tasks are often constructed by combining smaller task units together in time.

Below is an example of the generated tasks and the accompanying task graphs: 
### AS a reminder to add examples
iWISDM inherits several classes from COG (https://github.com/google/cog) to build task graphs. For convenience, we have pre-implemented several commonly used cognitive tasks in task_bank.py. 

For usage instructions, please refer to [Usage](#usage)

Additionally, for convenience, we have pre-generated four benchmarks of increased complexity level for evaluation of large multi-modal models. 

![Benchmark details for each level of complexity](https://github.com/BashivanLab/iWISDM/blob/main/benchmarking/param_table.png?raw=true)

These datasets can be generated from [/benchmarking](https://github.com/BashivanLab/iWISDM/tree/main/benchmarking) or downloaded: [iWISDM_benchsets.tar.gz](https://drive.google.com/file/d/1K-9AAJfvz6kiN3h9X2Rg0D88gJQ_rxSu/view?usp=sharing)

### For further details, please refer to (link to the paper)
### Paper link and a bibtex to copy for citations

# Usage
### Install Instructions

### Graphiz
- [Install graphiz on your machine](https://pygraphviz.github.io/documentation/stable/install.html)
### Poetry
#### Install Poetry
    curl -sSL https://install.python-poetry.org | python3 -

### Conda + Poetry
#### Create conda python environment
    conda create --name iwisdm python=3.11
#### Install packages
    poetry install

### ShapeNet Subset 
A large-scale repository of shapes represented by 3D CAD models of objects  [(Chang et. al. 2015)](https://arxiv.org/abs/1512.03012).
#### Pre-rendered Dataset Download
[shapenet_handpicked.tar.gz](https://drive.google.com/file/d/1is72QDjP6A6TA1mZLL3doYWaU08waAxm/view?usp=sharing) 

### Basic Usage
```python
# imports
from wisdom import make
from wisdom import read_write

# environment initialization
with open('../benchmarking/configs/high_complexity_all.json', 'r') as f:
    config = json.load(f)  # using pre-defined AutoTask configuration
env = make(env_id='ShapeNet')
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

### Acknowledgements
- COG - paper
- Thank funders?
