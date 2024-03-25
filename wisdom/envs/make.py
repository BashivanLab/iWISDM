from wisdom.core import Env
from wisdom.envs.registration import StimData, EnvSpec
from wisdom.envs.shapenet.shapenet_env import ShapeNetEnv


def make(
        env_id: str = 'ShapeNet',
        stim_data: StimData = None,
        env_spec: EnvSpec = None,
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
    @return:
    """
    assert env_id in env_dict, f"environment {env_id} not found in env_dict"
    env = env_dict[env_id](stim_data, env_spec)
    return env


env_dict = {
    'ShapeNet': ShapeNetEnv,
}
