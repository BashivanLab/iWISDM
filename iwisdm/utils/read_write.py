import json
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Tuple, List, Union

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from iwisdm.core import StimuliSet, StimData


@lru_cache(maxsize=20_000)
def _read_img_cached(fp: str, obj_size: Tuple[int, int]) -> NDArray:
    """Disk read + resize, cached on (path, size). random.choice(refs) in
    get_object() already happened by the time we're called, so caching here
    can't bias which object instance gets picked — it only skips the repeat
    disk read when the SAME resolved file comes up again across trials.
    Returned array is shared (not copied); safe because render_stim only
    slice-assigns it into a canvas, never mutates it in place."""
    image = cv2.imread(fp)
    if image is None:
        raise FileNotFoundError(f"cv2.imread returned None for {fp}")
    return cv2.resize(image, obj_size)


def read_img(fp: str, obj_size: Tuple[int, int], color_format='RGB'):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if color_format == 'RGB' else image
    return _read_img_cached(fp, obj_size)


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


def render_stim(canvas, obj, img_size, stim_data: StimData):
    """Render a static object.

    Args:
        canvas: numpy array of type int8 (img_size, img_size, 3). Modified in place.
          Importantly, opencv default is (B, G, R) instead of (R,G,B)
        obj: StaticObject instance
        img_size: int, image size.
        stim_data:
        mode:
    Returns:
        canvas: the modified frame, numpy array of type int8 (img_size, img_size, 3)
    """
    # Fixed specifications, see Space.sample()
    # when sampling, use a 1x1 grid, the most top-left position is (0.1, 0.1),
    # most bottom-right position is (0.9,0.9)
    # changing scaling requires changing space.sample)
    radius = int(0.25 * img_size)

    # Note that OpenCV color is (Blue, Green, Red)
    # TODO: change this from manual to grid_size
    center = [0, 0]
    if obj.location[0] < 0.5:
        center[0] = 56
    else:
        center[0] = 168
    if obj.location[1] < 0.5:
        center[1] = 56
    else:
        center[1] = 168
    # center = (int(obj.location[0] * img_size), int(obj.location[1] * img_size))

    x_offset, x_end = center[0] - radius, center[0] + radius
    y_offset, y_end = center[1] - radius, center[1] + radius
    obj_size = radius * 2
    shape_net_obj = stim_data.get_object(obj, (obj_size, obj_size))
    assert shape_net_obj.shape[:2] == (y_end - y_offset, x_end - x_offset)

    # note that openCV has x and y axis reversed
    canvas[y_offset:y_end, x_offset:x_end] = shape_net_obj
    return canvas


def render_stimset(stim_set: Union[List[StimuliSet], StimuliSet], canvas_size=224, stim_data: StimData = None):
    """
    Render a movie by epoch.

    @param stim_set: a StimuliSet instance or a list of them
    @param canvas_size: overall size of the rendered image
    @param stim_data: the stimuli dataset
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
                canvas = render_stim(canvas, obj.to_static()[0], canvas_size, stim_data)
            i_frame += 1
    return movie


def write_task(task, save_dir_fp, task_id=None):
    """
    Write the task to a json file

    @param task: a TemporalTask instance
    @param save_dir_fp: the directory to save the task
    @return:
    """
    if task_id:
        save_fp = os.path.join(save_dir_fp, f'task_{task_id}.json')
    else:
        save_fp = save_dir_fp
    info = task.to_json()
    with open(save_fp, 'w') as f:
        json.dump(info, f, indent=None)
    return


def _stack_uint8_hwc(imgs) -> np.ndarray:
    """
    Coerce trial frames to a contiguous uint8 (T, H, W, C) array.

    Accepts a list/tuple of per-frame numpy arrays or torch tensors, OR a single
    stacked array/tensor of shape (T, H, W, C) or (T, C, H, W). Channel ORDER is
    left untouched (iWISDM renders BGR — see render_stim / read_img); colour-order
    handling is the caller's job, per save_type.
    """
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().numpy()

    if isinstance(imgs, np.ndarray) and imgs.ndim == 4:
        frames = imgs
    else:
        frames = np.stack(
            [f.cpu().numpy() if isinstance(f, torch.Tensor) else np.asarray(f)
             for f in imgs],
            axis=0,
        )

    # (T, C, H, W) -> (T, H, W, C)
    if frames.shape[1] in (1, 3) and frames.shape[-1] not in (1, 3):
        frames = np.transpose(frames, (0, 2, 3, 1))

    if frames.dtype != np.uint8:
        if np.issubdtype(frames.dtype, np.floating):
            frames = frames * (255.0 if frames.size and frames.max() <= 1.0 else 1.0)
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(frames)


def write_trial(
        imgs,
        compo_info_dict,
        trial_fp: str,
        save_type: str = 'png',
) -> None:
    """
    Write a trial's frames and its task_info.json.

    @param imgs: per-frame images — a list/tuple of numpy arrays or torch tensors,
                 or a single stacked array/tensor of shape (T,H,W,C) or (T,C,H,W).
                 Expected in OpenCV BGR order, like the rest of this module.
    @param compo_info_dict: task-information dict, written as task_info.json.
    @param trial_fp: trial directory; frames go under <trial_fp>/frames.
    @param save_type: 'png' -> one epoch{i}.png per frame (cv2, legacy default), or
                      'pt'  -> all frames in one uint8 frames.pt (fast path).
    """
    frames_fp = os.path.join(trial_fp, 'frames')
    if os.path.exists(frames_fp):
        shutil.rmtree(frames_fp)
    os.makedirs(frames_fp)

    frames = _stack_uint8_hwc(imgs)  # (T, H, W, C) uint8, BGR

    if save_type == 'pt':
        # The dataloader reads via Image.fromarray, which assumes RGB. The legacy
        # cv2.imwrite path encoded BGR -> a true-colour RGB PNG, and PIL read it
        # back as RGB, so the model has always seen true RGB. Reverse BGR->RGB here
        # so frames.pt + Image.fromarray reproduces that byte-for-byte.
        if frames.shape[-1] == 3:
            frames = np.ascontiguousarray(frames[..., ::-1])
        torch.save(torch.from_numpy(frames), os.path.join(frames_fp, 'frames.pt'))

    elif save_type == 'png':
        # cv2.imwrite expects BGR; pass frames through unreversed (behaviour unchanged).
        for i in range(frames.shape[0]):
            cv2.imwrite(os.path.join(frames_fp, f'epoch{i}.png'), frames[i])

    else:
        raise ValueError(f"save_type must be 'png' or 'pt', got {save_type!r}")

    with open(os.path.join(frames_fp, 'task_info.json'), 'w') as f:
        json.dump(compo_info_dict, f, indent=None)
    return


def find_data_folder(data_folder: str = None):
    """
    In the project director data folder,
    find a subdirectory with stimuli images and csv file containing stimuli information.
    The subdirectory should have format:
    project/
        data/
            dataset_name/
                {train, val, test}/
                    ...
                meta.{csv, pkl}
    @param data_folder: the path to the data folder
    @return: the path to the data folder
    """
    if not data_folder:
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
