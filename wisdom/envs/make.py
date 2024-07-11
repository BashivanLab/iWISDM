from wisdom.core import Env
from wisdom.envs.registration import StimData, EnvSpec
from wisdom.envs.shapenet.shapenet_env import ShapeNetEnv
from wisdom.utils.read_write import find_data_folder


def make(
        env_id: str = 'ShapeNet',
        stim_data: StimData = None,
        env_spec: EnvSpec = None,
        dataset_fp: str = None,
) -> Env:
    """
    create an environment instance with
    1. specified stimuli dataset, the dataset directory contains the actual stimuli,
        and a meta info file that contains information about the stimuli
        see wisdom/envs/registration.py for more details on the stimuli dataset
    2. environment specifications that control how tasks are randomly generated
        see wisdom/envs/registration.py for more details on the environment specification
    @param env_id: the name of the environment
    @param stim_data: Data class that contains the stimuli dataset directory
    @param env_spec: Data class that contains the environment specification
    @param dataset_fp: the file path to the stimuli dataset
    @return:
    """
    assert env_id in env_dict, f"environment {env_id} not found in env_dict"
    env = env_dict[env_id]
    if stim_data is None:
        if dataset_fp is None:
            dataset_fp = find_data_folder()
        stim_data = env.init_stim_data(dataset_fp)
    if env_spec is None:
        env_spec = env.init_env_spec()
    env = env(stim_data, env_spec)
    return env


env_dict = {
    'ShapeNet': ShapeNetEnv,
}
