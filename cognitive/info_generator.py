"""
Classes for building temporal composite tasks
"""

import re
import numpy as np
from functools import partial
from collections import defaultdict
import os, shutil
import json
import cv2

import cognitive.constants as const
import cognitive.stim_generator as sg
import cognitive.task_generator as tg

from typing import Tuple, Dict, List, Set


class TaskInfoCompo(object):
    """
    Storage of composition information,
    including task_frame_info, task_example, task and objset
    correct usage: init objset and frame_info first, then generate TaskInfoCompo

    :param frame_info: FrameInfo
    :param task: task family
    """

    def __init__(self, task, frame_info):
        # combining with a second task should be implemented incrementally
        assert isinstance(task, tg.TemporalTask)

        self.task_objset = dict()
        self.tasks = [task]  # list of tasks in composition
        self.changed = list()
        self.frame_info = frame_info
        self.reset()

    def reset(self):
        if self.frame_info is None:
            objset = self.task.generate_objset()
            self.frame_info = FrameInfo(self.task, objset)

        self.task_objset[0] = self.frame_info.objset.copy()
        self.frame_info = self.frame_info
        self.tempo_dict = dict()

    def merge(self, new_task_info):
        """
        temporally combine the two tasks while reusing stimuli
        removed previous implementation where stimuli is not reused
        :param new_task_info: the new task to be temporally combined with the current task
        :return:
        """

        new_task = new_task_info.tasks[0]
        new_task_copy: tg.TemporalTask = new_task.copy()

        new_task_idx = len(self.tasks)
        start_frame_idx = self.frame_info.get_start_frame(new_task_info, relative_tasks={new_task_idx})

        # init new ObjSet to combine new and old
        objset = sg.ObjectSet(n_epoch=new_task_copy.n_frames)
        changed = False
        # change the necessary selects first
        for i, (old_frame, new_frame) in enumerate(zip(self.frame_info[start_frame_idx:], new_task_info.frame_info)):
            last_k = 'last%d' % (len(new_task_info.frame_info) - i - 1)
            # if there are objects in both frames, then update the new task's selects
            if old_frame.objs and new_frame.objs:
                # update the select such that it corresponds to the same object
                # checks how many selects and if there are enough objects for the selects
                filter_objs = new_task.reinit(new_task_copy, old_frame.objs, last_k)
                if filter_objs:
                    changed = True
                    for obj in filter_objs:
                        objset.add(obj=obj.copy(), epoch_now=new_task_copy.n_frames - 1, merge_idx=i)
                else:
                    raise RuntimeError('Unable to reuse')

        # get new objset for new task based on updated operators and merge
        if changed:
            # update objset based on existing objects, guess_objset resolves conflicts
            new_objset = new_task_copy.guess_objset(objset, new_task_copy.n_frames - 1)
            updated_fi = FrameInfo(new_task_copy, new_objset)  # initialize new frame info based on updated objset
            FrameInfo.update_relative_tasks(updated_fi, relative_tasks={new_task_idx})
            for i, (old_frame, new_frame) in enumerate(zip(self.frame_info[start_frame_idx:], updated_fi)):
                old_frame.compatible_merge(new_frame)  # update frame information

            self.changed.append((new_task_idx, new_task, new_task_copy))
            self.task_objset[new_task_idx] = new_objset  # update the per task objset dictionary
        else:
            for i, (old_frame, new_frame) in enumerate(
                    zip(self.frame_info[start_frame_idx:], new_task_info.frame_info)):
                old_frame.compatible_merge(new_frame)
            self.task_objset[new_task_idx] = new_task_info.task_objset[0]  # update the per task objset dictionary
        self.tasks.append(new_task)
        return

    def temporal_switch(self, task_info1, task_info2, switch_first=None):
        assert isinstance(task_info1, TaskInfoCompo)
        assert isinstance(task_info2, TaskInfoCompo)

        if any(task.is_bool_output() for task in [self.tasks[-1], task_info1.tasks[-1], task_info2.tasks[-1]]):
            raise ValueError('Switch tasks must have boolean outputs')

        self.tempo_dict['self'] = self.get_task_info_dict()[1]
        self.tempo_dict['task1'] = task_info1.get_task_info_dict()[1]
        self.tempo_dict['task2'] = task_info2.get_task_info_dict()[1]
        # current implementation of compo_task is by combining pre-generated objsets
        # so have to rely on main.py to generate both branches
        if not switch_first:
            answer = self.tempo_dict['self']['answers'][len(self.frame_info) - 1]
            if answer not in ['true', 'false']:
                raise ValueError('End of task is not boolean output')
            switch_first = True if answer == 'true' else False
        task_info = task_info1 if switch_first else task_info2

        self.frame_info.first_shareable = len(self.frame_info)
        self.merge(task_info)
        return

    def get_instruction_obj_info(self) -> Tuple[str, Dict[int, List]]:
        """ recompiles task instructions based on the composition.
        e.g. observe object1, delay, task1 instruction, delay, observe object2, task2 instruction.
        adds delay instructions during empty frames
        :return: changed task instruction and obj_info dictionary
        """
        obj_info = defaultdict(list)  # key: epoch, value: list of dictionary of object info
        count = 0
        cur = 0
        for epoch, objs in sorted(self.frame_info.objset.dict.items()):
            for i, obj in enumerate(objs):
                count += 1
                info = dict()
                info['count'] = count
                info['obj'] = obj
                info['tasks'] = set()  # tasks that involve the stim
                info['attended_attr'] = defaultdict(set)  # key: task, value: attribute of the stim that are relevant
                obj_info[epoch].append(info)

        # todo: uncomment below to allow instruction generation
        compo_instruction = ''
        if self.tempo_dict:
            compo_instruction += 'compo task 1: '
        # for each frame, find the tasks that use the stimuli. For each of these tasks,
        # compare the object in task_objset with obj, and rename lastk with ith object
        was_delay = False
        for epoch, frame in enumerate(self.frame_info):
            add_delay = True
            if obj_info[epoch]:  # if the frame contains stim/objects
                for info_dict in obj_info[epoch]:  # for each object info dict,
                    compo_instruction += f'observe object {info_dict["count"]}, '
                add_delay, was_delay = False, False

            for d in frame.description:
                if 'ending' in d:  # align object count with lastk for each task
                    task_idx = int(re.search(r'\d+', d).group())
                    task_q = str(self.tasks[task_idx])

                    # find epoch relative to ending task
                    lastks = [m for m in re.finditer('last\d+', task_q)]  # modify this for multiple stim per frame
                    for lastk in lastks:
                        k = int(re.search(r'\d+', lastk.group()).group())
                        i = epoch - k
                        relative_i = frame.relative_task_epoch_idx[task_idx] - k
                        match = self.compare_objs(obj_info[i], self.task_objset[task_idx].dict[relative_i])
                        if match:
                            task_q = re.sub(f'last{k}.*?object', f'object {match["count"]}', task_q)
                            match['tasks'].add(task_idx)
                            match['attended_attr'][task_idx] = match['attended_attr'][task_idx].union(
                                self.tasks[task_idx].get_relevant_attribute(f'last{k}'))
                            cur += 1
                        else:
                            raise RuntimeError('No match')
                    compo_instruction += task_q
                    add_delay, was_delay = False, False

                    if self.tempo_dict:
                        # in temporal switch, if at the end of the original task,
                        # then add the conditional texts in the instruction
                        if epoch == len(self.tempo_dict['self']['answers']) - 1:
                            compo_instruction += ' if end of compo task 1 is true, then do compo task 2: '
                            compo_instruction += self.object_add(self.tempo_dict['task1']['instruction'], cur)
                            compo_instruction += 'otherwise, do compo task 3: '
                            compo_instruction += self.object_add(self.tempo_dict['task2']['instruction'], cur)
                            return compo_instruction, obj_info
            if add_delay and not was_delay:
                compo_instruction += 'delay, '
                was_delay = True
        return compo_instruction, obj_info

    def get_target_value(self, examples: List[Dict]) -> List[str]:
        """
        get the correct response for each frame
        :param examples: list of dictionary containing the information about individual tasks
        :return: the response for each frame in list. if there is no response, then null is saved
        """
        answers = list()
        for frame in self.frame_info:
            ending = False
            for d in frame.description:  # see frame_info init
                if 'ending' in d:  # if the frame is the end of any individual task
                    task_idx = int(re.search(r'\d+', d).group())  # get which task is ending
                    answers.append(examples[task_idx]['answers'][0])  # add the task answer
                    ending = True
            if not ending:
                answers.append('null')
        return answers

    def get_task_info_dict(self, is_instruction=True, external_instruction=None) -> Tuple[List[Dict], Dict]:
        """
        get task examples
        :return: tuple of list of dictionaries containing information about the requested tasks
        and dictionary of the compositional task
        """
        per_task_info = list()
        for i, task in enumerate(self.tasks):
            per_task_info.append({
                'family': str(task.__class__.__name__),
                # saving an epoch explicitly is needed because
                # there might be no objects in the last epoch.
                'epochs': int(task.n_frames),
                'question': str(task),
                'objects': [o.dump() for o in self.task_objset[i]],
                'answers': [str(const.get_target_value(t)) for t in task.get_target(self.task_objset[i])],
                'first_shareable': int(task.first_shareable),
            })

        if is_instruction:
            comp_instruction, _ = self.get_instruction_obj_info()
        else:
            comp_instruction = external_instruction

        compo_info = {
            'epochs': int(len(self.frame_info)),
            'objects': [o.dump() for o in self.frame_info.objset],
            'instruction': comp_instruction,
            'answers': self.get_target_value(per_task_info)
        }
        return per_task_info, compo_info

    def generate_trial(self, img_size: int, fixation_cue: bool) -> Tuple[List[np.ndarray], List[Dict], Dict]:
        # add fixation cues to all frames except for task ending frames

        objset = self.frame_info.objset

        imgs = []
        for i, (epoch, frame) in enumerate(zip(sg.render(objset, img_size), self.frame_info)):
            if fixation_cue:
                if not any('ending' in description for description in frame.description):
                    sg.add_fixation_cue(epoch)
            imgs.append(epoch)
        per_task_info_dict, compo_info_dict = self.get_task_info_dict()
        return imgs, per_task_info_dict, compo_info_dict

    def write_trial(self, trial_fp: str, img_size: int = 224, fixation_cue: bool = False) -> None:
        """
        write the trial images, and save the task information in task_info.json

        @param trial_fp: the directory to write the frames, usually folder name is trial_i
        @param img_size: default 224x224 image size
        @param fixation_cue: if we add a cross in the middle of the image
        @return:
        """

        frames_fp = os.path.join(trial_fp, 'frames')
        if os.path.exists(frames_fp):
            shutil.rmtree(frames_fp)
        os.makedirs(frames_fp)

        imgs, per_task_info_dict, compo_info_dict = self.generate_trial(img_size, fixation_cue)
        for i, img_arr in enumerate(imgs):
            filename = os.path.join(frames_fp, f'epoch{i}.png')
            cv2.imwrite(filename, img_arr)

        per_task_info_dict, compo_info_dict = self.get_task_info_dict()

        filename = os.path.join(frames_fp, 'task_info.json')
        with open(filename, 'w') as f:
            json.dump(compo_info_dict, f, indent=4)

        """
        Commenting these out for faster data gen - LG, Sat Nov 25
        """
        # for i, task_example in enumerate(per_task_info_dict):
        #     filename = os.path.join(frames_fp, f'task{i} example')
        #     with open(filename, 'w') as f:
        #         json.dump(task_example, f, indent=4)

        # filename = os.path.join(frames_fp, 'frame_info')
        # with open(filename, 'w') as f:
        #     json.dump(self.frame_info.dump(), f, indent=4)
        return

    def get_changed_task_objset(self, changed_task: tg.TemporalTask) -> sg.ObjectSet:
        """
        reinitialize the ObjSet of the new ask after changing the task structure
        :param changed_task: the modified temporal task
        :return: new ObjSet according to the modified task graph
        """
        objset = changed_task.generate_objset()

        obj: sg.Object
        for obj in objset:
            for frame in self.frame_info:
                frame_obj: sg.Object
                for frame_obj in frame.objs:
                    if frame_obj.compare_attrs(obj, ['object', 'view_angle', 'category']):
                        obj.location = frame_obj.location
        return objset

    def __len__(self):
        # return number of tasks involved
        return len(self.frame_info)

    @property
    def n_epochs(self):
        return len(self.frame_info)

    @staticmethod
    def compare_objs(info_dicts, l2):
        for info_dict in info_dicts:
            obj1 = info_dict['obj']
            for obj2 in l2:
                if obj1.compare_attrs(obj2):
                    return info_dict
        return None

    @staticmethod
    def _add(match, x):
        # helper function for composite task instruction
        val = match.group()
        return re.sub(r'\d+', lambda m: str(int(m.group()) + x), val)

    def object_add(self, t: str, x: int):
        # finds all "object k" substrings, and substitute them with "object k+x"
        # see info_generator_test.testObjectAddOne() for example
        return re.sub(r'object \d+', partial(self._add, x=x), t)


class FrameInfo(object):
    def __init__(self, task: tg.TemporalTask, objset: sg.ObjectSet):
        """
        contains information about individual task's frames
        used for combining multiple temporal tasks, initialize with 1 task,
        stores each frame object in frame_list
        :param task: task_generator.TemporalTask object
        :param objset: objset related to the task
        """
        assert isinstance(objset, sg.ObjectSet)
        assert isinstance(task, tg.TemporalTask)
        if task.n_frames != objset.n_epoch:
            raise ValueError('Task epoch does not equal objset epoch')

        n_epochs = task.n_frames
        self.objset = objset
        self.frame_list = list()  # List of Frame objects
        self.n_epochs = n_epochs
        self.first_shareable = task.first_shareable
        # TODO: decide if task_question needs to be kept
        # task_question = str(task)
        # task_answers = [const.get_target_value(t) for t in task.get_target(objset)]

        for i in range(n_epochs):
            # keep track of which frames are the ending and starting of which tasks
            description = list()
            if i == self.n_epochs - 1:
                description.append("ending of task %d" % 0)
            else:
                if i == 0:
                    description.append("start of task %d" % 0)
            self.frame_list.append(self.Frame(fi=self,
                                              idx=i,
                                              relative_tasks={0},
                                              description=description))
        if objset:
            # iterate all objects in objset and add to each frame
            for obj in objset:
                # iterate over all frames that the objset belongs to
                for epoch in range(obj.epoch[0], obj.epoch[1]):
                    self.frame_list[epoch].objs.append(obj.copy())

        self.last_task = 0
        self.last_task_start = 0
        self.last_task_end = len(self.frame_list) - 1

    def dump(self) -> Dict[int, Dict]:
        # get information about the frame_info
        info = {i: dict() for i, frame in enumerate(self.frame_list)}
        for i, frame in enumerate(self.frame_list):
            info[i]['description'] = frame.description
            info[i]['relative_tasks'] = list(frame.relative_tasks)
            info[i]['relative_task_epoch_idx'] = frame.relative_task_epoch_idx
        return info

    def add_new_frames(self, i, relative_tasks):
        """
        add new empty frames and update objset and p during composition
        :param i: number of new frames
        :param relative_tasks: the tasks associated with the new frames
        :return: True if epoch increased
        """
        if i <= 0:
            return False
        for j in range(i):
            self.frame_list.append(self.Frame(fi=self,
                                              idx=len(self.frame_list),
                                              relative_tasks=relative_tasks
                                              ))
        self.objset.increase_epoch(self.objset.n_epoch + i)
        return True

    @staticmethod
    def update_relative_tasks(new_frame_info, relative_tasks):
        # update the relative_task info for each frame in the new task (from 0 to new task idx) after composition
        next_task_idx = list(relative_tasks)[0]
        for frame in new_frame_info:
            frame.relative_tasks = relative_tasks
            # new_task_info should only contain 1 task
            frame.relative_task_epoch_idx[next_task_idx] = frame.relative_task_epoch_idx.pop(0)
            for i, description in enumerate(frame.description):
                if 'start' in description:
                    frame.description[i] = f'start of task {next_task_idx}'
                if 'ending' in description:
                    frame.description[i] = f'ending of task {next_task_idx}'
        return

    def get_start_frame(self, new_task_info: TaskInfoCompo, relative_tasks: Set[int]):
        """
        randomly sample a starting frame to start merging and add new frames if needed

        this is done by checking length of both tasks, then starting first based on first_shareable

        if both start at the same frame, but new task ends earlier,
        then force the new task to start later by adding more empty frames to the old task.
        otherwise, if new task start frame is after existing task's start frame,
        then new task can end earlier than existing task

        Overall, task order is arranged such that new task appears or finishes after the existing task
        avoid overlapping response frames
        :param new_task_info: TaskInfoCompo object of the new task
        :param relative_tasks: set of task indices used by the new task, relative to the TaskInfoCompo
        :return: index of frame to begin merging with respect to existing frame_info
        """
        assert len(new_task_info.tasks) == 1

        first_shareable = self.first_shareable
        shareable_frames = self.frame_list[first_shareable:]
        new_task_len = new_task_info.n_epochs
        new_first_shareable = new_task_info.tasks[0].first_shareable

        self.update_relative_tasks(new_task_info.frame_info, relative_tasks)

        if len(shareable_frames) == 0:
            # queue, add frames and start merging
            self.first_shareable = len(self.frame_list)
            self.add_new_frames(new_task_len, relative_tasks)

            self.last_task_end = len(self.frame_list) - 1
            self.last_task = list(relative_tasks)[0]
            self.first_shareable = new_first_shareable + first_shareable
            return first_shareable
        else:
            # add more frames
            if len(shareable_frames) == new_task_len:
                self.add_new_frames(1, relative_tasks)
                first_shareable += 1
            else:
                self.add_new_frames(new_task_len - len(shareable_frames), relative_tasks)

            shareable_frames = self.frame_list[first_shareable:]
            # first check if start of alignment
            # if the alignment starts with start of existing task, then check if new task ends earlier
            aligned = False
            while not aligned:
                # inserting t3 into T = {t1,t2} (all shareable) in cases where t1 ends before t2 but appears before t2,
                # last_task_start is after t1, last_task_end is after t1.
                # suppose shareable_frames[new_task_len] is the index of end of t1
                # then shift to the right, at next loop, t2 ends, then add new frame

                # check where the response frame of the new task would be in shareable_frames
                # if the response frames are overlapping, then shift alignment to "right"
                if first_shareable + new_task_len - 1 == self.last_task_end:
                    first_shareable += 1
                # if the sample frames are overlapping, then check if the new task ends before the last task
                # if so, then shift starting frame
                elif first_shareable == self.last_task_start \
                        and first_shareable + new_task_len - 1 <= self.last_task_end:
                    first_shareable += 1
                else:
                    aligned = True

                if len(shareable_frames) < new_task_len:
                    self.add_new_frames(1, relative_tasks)
                shareable_frames = self.frame_list[first_shareable:]

            self.last_task_end = first_shareable + new_task_len - 1
            self.last_task = list(relative_tasks)[0]
            self.last_task_start = first_shareable

            self.first_shareable = new_first_shareable + first_shareable
            return first_shareable

    def __len__(self):
        return len(self.frame_list)

    def __iter__(self):
        return self.frame_list.__iter__()

    def __getitem__(self, item):
        return self.frame_list.__getitem__(item)

    class Frame(object):
        # frame object within frame_info list, used to merge frames from different tasks
        def __init__(self, fi, idx: int, relative_tasks: Set[int], description: List[str] = None,
                     objs: List[sg.Object] = None):
            """
            :param fi: the frame info object this frame belongs to
            :param idx: the index of the frame
            :param relative_tasks: which tasks this frame belongs to
            :param description: information about the frame
            :param objs: the list of objects in this frame
            """
            assert isinstance(fi, FrameInfo)

            self.fi = fi
            self.idx = idx
            self.relative_tasks = relative_tasks
            self.description = description if description else list()
            self.objs = objs if objs else list()

            self.relative_task_epoch_idx = dict()
            for task in self.relative_tasks:
                self.relative_task_epoch_idx[task] = self.idx

        def compatible_merge(self, new_frame):
            assert isinstance(new_frame, FrameInfo.Frame)

            self.relative_tasks = self.relative_tasks | new_frame.relative_tasks  # union of two sets of tasks
            self.description = self.description + new_frame.description  # append new descriptions

            for new_obj in new_frame.objs:
                self.fi.objset.add(new_obj.copy(), len(self.fi) - 1, merge_idx=self.idx)  # update objset
            self.objs = self.fi.objset.dict[self.idx].copy()

            # update the dictionary for relative_task_epoch
            temp = self.relative_task_epoch_idx.copy()
            temp.update(new_frame.relative_task_epoch_idx)
            self.relative_task_epoch_idx = temp

        def __str__(self):
            return 'frame: ' + str(self.idx) + ', relative tasks: ' + \
                ','.join([str(i) for i in self.relative_tasks]) \
                + ' objects: ' + ','.join([str(o) for o in self.objs])
