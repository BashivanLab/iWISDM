"""
Classes for building temporal composite tasks
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Tuple, Dict, List, Optional, Set

import iwisdm.envs.shapenet.registration as env_reg
from iwisdm.envs.shapenet.registration import SNStimData
import iwisdm.envs.shapenet.stim_generator as sg
import iwisdm.envs.shapenet.task_generator as tg
from iwisdm.utils.read_write import add_cross, render_stimset


class Frame(object):
    """A single frame in a composed task timeline.

    Frames are passive containers; all mutation that involves the shared
    objset goes through FrameInfo (see FrameInfo.merge_frame).
    Task-boundary and distractor metadata is stored structurally
    (task_starts, task_ends, distractor_attrs) instead of as description
    strings; use the description property for a human-readable form.
    """

    def __init__(
            self,
            idx: int,
            relative_tasks: Set[int],
            task_starts: Set[int] = None,
            task_ends: Set[int] = None,
            distractor_attrs: List[str] = None,
            objs: List[sg.Object] = None
    ):
        """
        :param idx: the index of the frame
        :param relative_tasks: which tasks this frame belongs to
        :param task_starts: tasks whose first frame is this frame
        :param task_ends: tasks whose response frame is this frame
        :param distractor_attrs: attributes that distinguish distractors in this frame
        :param objs: the list of objects in this frame
        """
        self.idx = idx
        self.relative_tasks = set(relative_tasks)
        self.task_starts = set(task_starts) if task_starts else set()
        self.task_ends = set(task_ends) if task_ends else set()
        self.distractor_attrs = list(distractor_attrs) if distractor_attrs else list()
        self.objs = objs if objs else list()

    @property
    def description(self) -> List[str]:
        # human-readable form of the frame markers
        desc = [f'start of task {i}' for i in sorted(self.task_starts)]
        desc += [f'ending of task {i}' for i in sorted(self.task_ends)]
        if self.distractor_attrs:
            desc += [f'distractor with different {attr}' for attr in self.distractor_attrs]
        elif -1 in self.relative_tasks:
            desc.append('distractor')
        return desc

    def copy(self):
        return Frame(
            idx=self.idx,
            relative_tasks=set(self.relative_tasks),
            task_starts=set(self.task_starts),
            task_ends=set(self.task_ends),
            distractor_attrs=list(self.distractor_attrs),
            objs=[o.copy() for o in self.objs],
        )

    def __str__(self):
        return 'frame: ' + str(self.idx) + ', relative tasks: ' + \
            ','.join([str(i) for i in self.relative_tasks]) \
            + ' objects: ' + ','.join([str(o) for o in self.objs])


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
        # task_answers = [env_reg.get_target_value(t) for t in task.get_target(objset)]

        for i in range(n_epochs):
            # keep track of which frames are the ending and starting of which tasks
            task_starts, task_ends = set(), set()
            if i == self.n_epochs - 1:
                task_ends.add(0)
            elif i == 0:
                task_starts.add(0)
            self.frame_list.append(
                Frame(
                    idx=i,
                    relative_tasks={0},
                    task_starts=task_starts,
                    task_ends=task_ends,
                )
            )
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
        info = dict()
        for i, frame in enumerate(self.frame_list):
            info[i] = {
                'description': frame.description,
                'relative_tasks': list(frame.relative_tasks),
            }
        return info

    def copy(self):
        """
        :return: deep copy of the timeline, sharing no mutable state with self
        """
        new = FrameInfo.__new__(FrameInfo)
        new.objset = self.objset.copy()
        new.frame_list = [frame.copy() for frame in self.frame_list]
        new.n_epochs = self.n_epochs
        new.first_shareable = self.first_shareable
        new.last_task = self.last_task
        new.last_task_start = self.last_task_start
        new.last_task_end = self.last_task_end
        return new

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
            self.frame_list.append(Frame(idx=len(self.frame_list),
                                         relative_tasks=relative_tasks
                                         ))
        self.objset.increase_epoch(self.objset.n_epoch + i)
        return True

    def merge_frame(self, idx: int, new_frame: Frame):
        """
        merge a frame from another task into the frame at idx,
        combining task markers and adding the new frame's objects to the shared objset
        :param idx: index of the frame in this timeline to merge into
        :param new_frame: the frame from the task being merged in
        """
        frame = self.frame_list[idx]
        frame.relative_tasks |= new_frame.relative_tasks  # union of the two sets of tasks
        frame.task_starts |= new_frame.task_starts
        frame.task_ends |= new_frame.task_ends
        frame.distractor_attrs += new_frame.distractor_attrs

        for new_obj in new_frame.objs:
            self.objset.add(new_obj.copy(), len(self.frame_list) - 1, merge_idx=idx)
        if len(self.objset.dict[idx]) > 1:
            raise RuntimeError(f'More than 1 object in frame {self.objset.dict[idx]}')
        frame.objs = self.objset.dict[idx].copy()

    def add_distractor(self, distractor: sg.Object, frame_idx: int, diff_attr: str = None):
        """
        add a distractor object to a frame
        :param distractor: the distractor object
        :param frame_idx: which frame to add it to
        :param diff_attr: the attribute distinguishing the distractor from the frame's stimulus,
        if it was constructed to differ in one
        """
        frame = self.frame_list[frame_idx]
        frame.objs.append(distractor)
        frame.relative_tasks.add(-1)
        if diff_attr:
            frame.distractor_attrs.append(diff_attr)
        self.objset.add(
            obj=distractor,
            epoch_now=len(self.frame_list) - 1,
            delete_if_can=True,
            merge_idx=frame_idx
        )
        return

    @staticmethod
    def update_relative_tasks(new_frame_info, relative_tasks):
        # update the relative_task info for each frame in the new task (from 0 to new task idx) after composition
        next_task_idx = list(relative_tasks)[0]
        for frame in new_frame_info:
            if frame.relative_tasks != {0}:
                # new_task_info should only contain 1 task, indexed 0
                raise RuntimeError(
                    f'expected a single-task frame info, got tasks {frame.relative_tasks}')
            frame.relative_tasks = set(relative_tasks)
            if 0 in frame.task_starts:
                frame.task_starts = {next_task_idx}
            if 0 in frame.task_ends:
                frame.task_ends = {next_task_idx}
        return

    @staticmethod
    def find_start_frame(
            first_shareable: int,
            n_frames: int,
            new_task_len: int,
            last_task_start: int,
            last_task_end: int,
    ) -> Tuple[int, int]:
        """
        pure placement logic for merging a new task into an existing timeline

        task order is arranged such that the new task appears or finishes after
        the existing task, avoiding overlapping response frames:
        the new task's response frame must not coincide with the previous task's
        response frame, and if both tasks start at the same frame, the new task
        must end after the previous task

        :param first_shareable: index of the first frame the new task may share
        :param n_frames: current number of frames in the timeline
        :param new_task_len: number of frames of the new task
        :param last_task_start: start frame index of the previously merged task
        :param last_task_end: response frame index of the previously merged task
        :return: (start frame index for merging, total frames the timeline needs)
        """
        start = first_shareable
        n_shareable = n_frames - start
        if n_shareable == 0:
            # queue the new task after all existing frames
            return start, start + new_task_len
        if n_shareable == new_task_len:
            # an exact fit would place the new response frame on the timeline's
            # last frame; shift right so response frames cannot coincide
            start += 1
            n_frames += 1
        elif n_shareable < new_task_len:
            n_frames += new_task_len - n_shareable
        while (start + new_task_len - 1 == last_task_end
               or (start == last_task_start and start + new_task_len - 1 <= last_task_end)):
            start += 1
            if n_frames - start < new_task_len:
                n_frames += 1
        return start, n_frames

    def get_start_frame(self, new_task_info: TaskInfoCompo, relative_tasks: Set[int]):
        """
        find the starting frame for merging and add new frames if needed

        placement is computed by find_start_frame; this method materializes the
        needed frames and updates the timeline bookkeeping
        :param new_task_info: TaskInfoCompo object of the new task
        :param relative_tasks: set of task indices used by the new task, relative to the TaskInfoCompo
        :return: index of frame to begin merging with respect to existing frame_info
        """
        assert len(new_task_info.tasks) == 1
        self.update_relative_tasks(new_task_info.frame_info, relative_tasks)

        new_task_len = new_task_info.n_epochs
        start, n_frames = self.find_start_frame(
            self.first_shareable, len(self.frame_list), new_task_len,
            self.last_task_start, self.last_task_end,
        )
        self.add_new_frames(n_frames - len(self.frame_list), relative_tasks)

        self.last_task = list(relative_tasks)[0]
        self.last_task_start = start
        self.last_task_end = start + new_task_len - 1
        self.first_shareable = new_task_info.tasks[0].first_shareable + start
        return start

    def __len__(self):
        return len(self.frame_list)

    def __iter__(self):
        return self.frame_list.__iter__()

    def __getitem__(self, item):
        return self.frame_list.__getitem__(item)

@dataclass
class ComposedTask:
    """Record of one task within a composition.

    task: the instruction-facing task; its selects are soft-updated when
        stimuli are reused during merging
    bound_task: copy of task whose selects were hard-bound to the reused
        stimuli in order to regenerate the objset; None if the merge did
        not reuse any stimuli
    objset: the stimuli instantiated for this task alone
    start_frame: index of the task's first frame in the composed timeline
    """
    task: tg.TemporalTask
    bound_task: Optional[tg.TemporalTask]
    objset: sg.ObjectSet
    start_frame: int


class TaskInfoCompo(object):
    """
    Storage of composition information

    each task in the composition is tracked by a ComposedTask record;
    the shared timeline is tracked by frame_info

    :param task: the first task of the composition
    :param frame_info: FrameInfo of the task, built from the task and its objset;
        if None, the task is instantiated with a fresh objset
    """

    def __init__(self, task: tg.TemporalTask, frame_info: FrameInfo = None):
        # combining with a second task should be implemented incrementally
        if frame_info is None:
            frame_info = FrameInfo(task, task.generate_objset())

        self.frame_info = frame_info
        self.composed = [ComposedTask(
            task=task,
            bound_task=None,
            objset=frame_info.objset.copy(),
            start_frame=0,
        )]
        self.tempo_dict = dict()

    @property
    def tasks(self) -> List[tg.TemporalTask]:
        # instruction-facing tasks, indexed by their position in the composition
        return [record.task for record in self.composed]

    @property
    def task_objset(self) -> Dict[int, sg.ObjectSet]:
        # per-task objsets, indexed by the task's position in the composition
        return {i: record.objset for i, record in enumerate(self.composed)}

    def copy(self):
        """
        :return: deep copy of the composition, sharing no mutable state with self
        """
        new = TaskInfoCompo.__new__(TaskInfoCompo)
        new.frame_info = self.frame_info.copy()
        new.composed = [ComposedTask(
            task=record.task.copy(),
            bound_task=record.bound_task.copy() if record.bound_task else None,
            objset=record.objset.copy(),
            start_frame=record.start_frame,
        ) for record in self.composed]
        new.tempo_dict = deepcopy(self.tempo_dict)
        return new

    def merge(self, new_task_info):
        """
        temporally combine the two tasks while reusing stimuli

        a private copy of new_task_info is merged; the argument is not modified
        :param new_task_info: the new task to be temporally combined with the current task
        :return:
        """
        new_task_info = new_task_info.copy()
        new_task = new_task_info.tasks[0]
        bound_task: tg.TemporalTask = new_task.copy()

        new_task_idx = len(self.composed)
        start_frame_idx = self.frame_info.get_start_frame(new_task_info, relative_tasks={new_task_idx})

        # init new ObjSet to combine new and old
        objset = sg.ObjectSet(n_epoch=bound_task.n_frames)
        changed = False
        # change the necessary selects first
        for i, (old_frame, new_frame) in enumerate(zip(self.frame_info[start_frame_idx:], new_task_info.frame_info)):
            # if there are objects in both frames, then update the new task's selects
            if old_frame.objs and new_frame.objs:
                # update the select such that it corresponds to the same object
                # checks how many selects and if there are enough objects for the selects
                last_k = 'last%d' % (len(new_task_info.frame_info) - i - 1)
                filter_objs = new_task.reinit(bound_task, old_frame.objs, last_k)
                if not filter_objs:
                    raise RuntimeError('Unable to reuse')
                changed = True
                for obj in filter_objs:
                    objset.add(obj=obj.copy(), epoch_now=bound_task.n_frames - 1, merge_idx=i)

        if changed:
            # bound_task has its task graph (Selects) changed to use the old stimuli;
            # regenerate its objset from them, guess_objset resolves conflicts
            new_objset = bound_task.guess_objset(objset, bound_task.n_frames - 1)
            merge_source = FrameInfo(bound_task, new_objset)  # new frame info based on the updated objset
            FrameInfo.update_relative_tasks(merge_source, relative_tasks={new_task_idx})
        else:
            new_objset = new_task_info.task_objset[0]
            merge_source = new_task_info.frame_info

        for i, new_frame in enumerate(merge_source):
            self.frame_info.merge_frame(start_frame_idx + i, new_frame)

        self.composed.append(ComposedTask(
            task=new_task,
            bound_task=bound_task if changed else None,
            objset=new_objset,
            start_frame=start_frame_idx,
        ))
        return

    def temporal_switch(self, task_info1, task_info2, switch_first=None):
        assert isinstance(task_info1, TaskInfoCompo)
        assert isinstance(task_info2, TaskInfoCompo)

        if any(task.is_bool_output() for task in [self.tasks[-1], task_info1.tasks[-1], task_info2.tasks[-1]]):
            raise ValueError('Switch tasks must have boolean outputs')

        self.tempo_dict['self'] = self.get_task_info_dict()[1]
        self.tempo_dict['task1'] = task_info1.get_task_info_dict()[1]
        self.tempo_dict['task2'] = task_info2.get_task_info_dict()[1]

        if not switch_first:
            answer = self.tempo_dict['self']['answers'][len(self.frame_info) - 1]
            if answer not in ['true', 'false']:
                raise ValueError('End of task is not boolean output')
            switch_first = True if answer == 'true' else False
        task_info = task_info1 if switch_first else task_info2

        self.frame_info.first_shareable = len(self.frame_info)
        self.merge(task_info)
        return

    def get_obj_info_dict(self):
        obj_info = defaultdict(list)  # key: epoch, value: list of dictionary of object info
        count = 0
        for epoch, objs in sorted(self.frame_info.objset.dict.items()):
            num_objs = len(objs)
            if num_objs > 0:
                count += 1
            for obj in objs:
                # if not obj.deletable or num_objs == 1:
                #     # remove this if instruction says observe distractor with add_attr
                #     count += 1
                info = dict()
                info['count'] = count
                info['obj'] = obj
                info['tasks'] = set()  # tasks that involve the stim
                info['attended_attr'] = defaultdict(set)  # key: task, value: attribute of the stim that are relevant
                obj_info[epoch].append(info)
        return obj_info

    def get_instruction_obj_info(self) -> Tuple[str, Dict[int, List]]:
        """ recompiles task instructions based on the composition.
        e.g. observe object1, delay, task1 instruction, delay, observe object2, task2 instruction.
        adds delay instructions during empty frames
        :return: changed task instruction and obj_info dictionary
        """
        obj_info = self.get_obj_info_dict()
        cur = 0

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
                if len(obj_info[epoch]) == 1:
                    info_dict = obj_info[epoch][0]
                    compo_instruction += f'observe object {info_dict["count"]}, '
                else:  # if there are multiple stimuli in the frame (i.e. distractor)
                    distractor_attr = frame.distractor_attrs
                    if not distractor_attr:
                        task_str = ''
                        obj_str = ''
                        for i, task in enumerate(self.tasks):
                            task_str += f'task {i}: {str(task)}, '
                        for i, o in enumerate(obj_info[epoch]):
                            o_str = o['obj']
                            obj_str += f'obj {i}: {str(o_str)}, '
                        raise RuntimeError(
                            f'Found more than 1 object at epoch {epoch}, '
                            f'but no distractor description was found. '
                            f'This behaviour is unexpected.\n'
                            f'Task string: {task_str}\n'
                            f'Object string: {obj_str}\n'
                        )

                    for info_dict in obj_info[epoch]:
                        if info_dict['obj'].deletable:
                            # remove this if instruction says observe distractor with add_attr
                            continue
                        add_attr, loc_attr = '', ''
                        for attr in distractor_attr:
                            if attr == 'location':
                                loc_attr += f' at {attr}: {getattr(info_dict["obj"], attr)}'
                            else:
                                add_attr += f' with {attr}: {getattr(info_dict["obj"], attr)}'
                        compo_instruction += f'observe object {info_dict["count"]}{add_attr}{loc_attr}, '
                add_delay, was_delay = False, False

            for task_idx in sorted(frame.task_ends):  # align object count with lastk for each ending task
                task_instruction = str(self.tasks[task_idx])

                # find epoch relative to ending task
                lastks = [m for m in re.finditer(r'last\d+', task_instruction)]
                # TODO: modify this for multiple stim per frame
                for lastk in lastks:
                    k = int(re.search(r'\d+', lastk.group()).group())
                    i = epoch - k
                    # frame index relative to the ending task's own timeline
                    relative_i = epoch - self.composed[task_idx].start_frame - k
                    match = self.compare_objs(obj_info[i], self.task_objset[task_idx].dict[relative_i])
                    if match:
                        task_instruction = re.sub(
                            f'last{k}.*?object',
                            f'object {match["count"]}',
                            task_instruction
                        )
                        match['tasks'].add(task_idx)
                        match['attended_attr'][task_idx] = match['attended_attr'][task_idx].union(
                            self.tasks[task_idx].get_relevant_attribute(f'last{k}'))
                        cur += 1
                    else:
                        raise RuntimeError('No match')
                compo_instruction += (task_instruction + ' ') \
                    if epoch != len(self.frame_info) - 1 else task_instruction
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
            if frame.task_ends:  # if the frame is the end of any individual task
                for task_idx in sorted(frame.task_ends):
                    answers.append(examples[task_idx]['answers'][0])  # add the task answer
            else:
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
                'answers': [str(env_reg.get_target_value(t)) for t in task.get_target(self.task_objset[i])],
                'first_shareable': int(task.first_shareable),
            })

        if is_instruction:
            comp_instruction, obj_info = self.get_instruction_obj_info()
            objs = list()
            for epoch, info_dicts in obj_info.items():
                for info_dict in sorted(info_dicts, key=lambda x: x['count']):
                    info_dict['obj'] = info_dict['obj'].dump()
                    info_dict['tasks'] = list(info_dict['tasks'])
                    info_dict['attended_attr'] = {k: list(v) for k, v in info_dict['attended_attr'].items()}
                    objs.append(info_dict)
        else:
            comp_instruction = external_instruction
            objs = [o.dump() for o in self.frame_info.objset]

        compo_info = {
            'epochs': int(len(self.frame_info)),
            'objects': objs,
            'instruction': comp_instruction,
            'answers': self.get_target_value(per_task_info)
        }
        return per_task_info, compo_info

    def add_distractor_time(self, n_distractor: int, stim_data: SNStimData):
        """
        add distractors to the empty frames of a trial
        in the simplest case, if original task is:
            observe object 1, delay, observe object 2,
            attr of object 1 same as attr of object 2?
        the new task will be:
            observe object 1, observe object 2, observe object 3,
            attr of object 1 same as attr of object 3?
        @param n_distractor: how many distractors to add
        (bounded by the number of empty frames in the trial)
        @return:
        """
        empty_frames = [i for i, frame in enumerate(self.frame_info) if not frame.objs]
        if n_distractor > len(empty_frames):
            n_distractor = len(empty_frames)
        frames_to_add = sorted(random.sample(empty_frames, n_distractor))
        for i in frames_to_add:
            self.frame_info.add_distractor(
                distractor=sg.Object(when=i, deletable=True),
                frame_idx=i
            )
        return

    def add_distractor_frame(self, n_distractor: int, stim_data: SNStimData):
        """
        add distractors to the sample frames of a trial
        in the simplest case, if original task is:
            observe object 1, observe object 2,
            attr of object 1 same as attr of object 2?
        the new task will be:
            observe (object 1 with attr x), observe object 2,
            diff_attr of object 1 same as diff_attr of object 3?
        e.g. attr is category, x is chair, y is car, diff_attr is location, so the new task instruction is:
            observe object 1 with category chair, observe object 3,
            location of object 1 same as location of object 3?

        @param n_distractor: how many distractors to add
        (bounded by how many tasks, and how many sample frames in each task)
        @return:
        """
        stim_frames = [i for i, frame in enumerate(self.frame_info) if len(frame.objs) == 1]
        relevant_frames = [i for i in stim_frames if -1 not in self.frame_info[i].relative_tasks]
        if n_distractor > len(relevant_frames):
            n_distractor = len(relevant_frames)

        frames_to_add = sorted(random.sample(relevant_frames, n_distractor))
        for i in frames_to_add:
            frame = self.frame_info[i]
            attrs = set()
            for task_idx in frame.relative_tasks:
                task = self.tasks[task_idx]
                task_epoch = i - self.composed[task_idx].start_frame  # epoch relative to the task's own timeline
                attrs = attrs.union(task.get_relevant_attribute(f'last{task.n_frames - task_epoch - 1}'))
            attr_new_object = list()
            other_attrs = stim_data.ALL_ATTRS - attrs
            if not other_attrs:  # all attributes are used, cannot add distractor with different attribute
                continue

            for attr_type in other_attrs | {'location'}:
                existing_obj = frame.objs[0]
                attr_new_object.append(sg.another_attr(getattr(existing_obj, attr_type)))

            self.frame_info.add_distractor(
                distractor=sg.Object(attrs=attr_new_object, when=i, deletable=True),
                frame_idx=i,
                diff_attr=other_attrs.pop()
            )
        return

    def generate_trial(
            self,
            canvas_size: int,
            fixation_cue: bool,
            cue_on_action: bool,
            stim_data: SNStimData,
            return_objset: bool = False,
            add_distractor_frame: int = 0,
            add_distractor_time: int = 0,
    ):
        # TODO: return copy of objset, not add distractor in place
        if add_distractor_frame > 0:
            self.add_distractor_frame(add_distractor_frame, stim_data)

        if add_distractor_time > 0:
            self.add_distractor_time(add_distractor_time, stim_data)

        objset = self.frame_info.objset
        per_task_info_dict, compo_info_dict = self.get_task_info_dict()

        imgs = []
        for i, (epoch, frame) in enumerate(zip(render_stimset(objset, canvas_size, stim_data), self.frame_info)):
            if fixation_cue:
                end_of_task = bool(frame.task_ends)
                if cue_on_action:
                    # add fixation cues to all frames except for task ending frames
                    if end_of_task:
                        epoch = add_cross(epoch)
                else:
                    # add fixation cues to all task ending frames
                    if not end_of_task:
                        epoch = add_cross(epoch)
            imgs.append(epoch)

        if return_objset:
            # print('returning objset')
            return imgs, per_task_info_dict, compo_info_dict, objset
        return imgs, per_task_info_dict, compo_info_dict

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
            if not obj1.deletable:
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
