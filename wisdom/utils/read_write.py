import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Union

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


def write_trial():
    return
