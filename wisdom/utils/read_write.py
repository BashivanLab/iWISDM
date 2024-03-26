import os
from typing import Tuple, List, Union
import shutil
from pathlib import Path

import json
import cv2
import numpy as np
from numpy.typing import NDArray

from wisdom.core import StimuliSet


def read_img(fp: str, obj_size: Tuple[int, int], color_format='RGB'):
    image = cv2.imread(fp)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if color_format == 'RGB' else image
    object_arr = cv2.resize(image, obj_size)
    return object_arr


def add_cross(canvas: NDArray, cue_size: float = 0.05):
    """
    Add a cross to the center of the canvas
    @param canvas: the image array
    @param cue_size: the size of the cross relative to the canvas size
    @return:
    """
    img_size = canvas.shape[0]
    radius = int(cue_size * img_size)
    center = (canvas.shape[0] // 2, canvas.shape[1] // 2)
    thickness = int(0.02 * img_size)
    cv2.line(canvas, (center[0] - radius, center[1]),
             (center[0] + radius, center[1]), (255, 255, 255), thickness)
    cv2.line(canvas, (center[0], center[1] - radius),
             (center[0], center[1] + radius), (255, 255, 255), thickness)
    return canvas


def render_stimset(stim_set: Union[List[StimuliSet], StimuliSet], canvas_size=224, mode: str = 'train'):
    """
    Render a movie by epoch.

    @param stim_set: a StimuliSet instance or a list of them
    @param canvas_size: overall size of the rendered image
    @param mode: dataset split
    @return: numpy array (n_time, img_size, img_size, 3)
    """
    if not isinstance(stim_set, list):
        stim_set = [stim_set]

    n_objset = len(stim_set)
    n_epoch_max = max([s.n_epoch for s in stim_set])

    # It's faster if use uint8 here, but later conversion to float32 seems slow
    movie = np.zeros((n_objset * n_epoch_max, canvas_size, canvas_size, 3), np.uint8)

    i_frame = 0
    for s in stim_set:
        for epoch_now in range(n_epoch_max):
            canvas = movie[i_frame:i_frame + 1, ...]  # return a view
            canvas = np.squeeze(canvas, axis=0)

            subset = s.select_now(epoch_now)
            for obj in subset:
                obj.render(canvas, canvas_size, mode)
            i_frame += 1
    return movie


def write_trial(imgs, compo_info_dict, trial_fp: str) -> None:
    """
    write the trial images, and save the task information in task_info.json

    @param imgs: a list of numpy arrays, each array is an image
    @param compo_info_dict: a dictionary containing task information
    @param trial_fp: the directory to write the frames, usually folder name is trial_i
    @return:
    """

    frames_fp = os.path.join(trial_fp, 'frames')
    if os.path.exists(frames_fp):
        shutil.rmtree(frames_fp)
    os.makedirs(frames_fp)

    for i, img_arr in enumerate(imgs):
        filename = os.path.join(frames_fp, f'epoch{i}.png')
        cv2.imwrite(filename, img_arr)

    filename = os.path.join(frames_fp, 'task_info.json')
    with open(filename, 'w') as f:
        json.dump(compo_info_dict, f, indent=4)
    return


def find_data_folder():
    """
    In the project director data folder,
    find a subdirectory with stimuli images and csv file containing stimuli information.
    The subdirectory should have format:
    data/
        dataset_name/
            {train, val, test}/
                ...
            meta.{csv, pkl}
    @return: the path to the data folder
    """
    data_folder = os.path.join(os.getcwd(), 'data')
    if os.path.isdir(data_folder):
        for sub_dir in os.listdir(data_folder):
            dir_path = os.path.join(data_folder, sub_dir)
            if os.path.isdir(dir_path):
                if list(Path(dir_path).rglob('*.csv')) or list(Path(dir_path).rglob('*.pkl')):
                    if (os.path.isdir(os.path.join(dir_path, 'train')) or os.path.isdir(
                            os.path.join(dir_path, 'validation')) or os.path.isdir(os.path.join(dir_path, 'test'))):
                        return dir_path
    raise ValueError(f'No dataset found in data folder {data_folder}. Specify the dataset path.')
