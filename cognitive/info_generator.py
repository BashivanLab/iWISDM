""" Classes for building composite tasks
"""
import numpy as np

import cognitive.constants as const
import cognitive.stim_generator as sg
import cognitive.task_generator as tg
import re


class ReplaceLastK(object):
    def __init__(self, task_objset: sg.ObjectSet, objs: list):
        self.count = set()
        obj: sg.Object

        for task_obj in task_objset:
            matches = set()
            for i, obj in enumerate(objs):
                if obj.compare_attrs(task_obj,
                                     attrs=['object', 'view_angle', 'category', 'loc']):
                    matches.add(i)
            while matches:
                match = max(matches)
                self.count.add(match)
                matches.remove(match)

    def __call__(self, match):
        idx = self.count.pop()
        return "object {}".format(idx + 1)


class TaskInfoCompo(object):
    """
    Storage of composition information,
    including task_frame_info, task_example, task and objset
    correct usage: generate frame_info first, then generate TaskInfoCompo
    :param frame_info: FrameInfo
    :param task: task family
    """

    def __init__(self, task, frame_info=None):
        # combining with a second task should be implemented incrementally
        assert isinstance(task, tg.TemporalTask)

        self.task_objset = dict()
        self.tasks = [task]
        self.changed = list()
        if frame_info is None:
            print('Alert! Correct usage: generate frame_info first, then generate TaskInfoCompo')
            if task.n_distractors is None or task.avg_mem is None:
                objset = task.generate_objset()
            else:
                objset = task.generate_objset(task.n_distractors, task.avg_mem)
            frame_info = FrameInfo(task, objset)

        self.task_objset[0] = frame_info.objset.copy()
        self.frame_info = frame_info

    def __len__(self):
        # return number of tasks involved
        return len(self.frame_info)

    def __str__(self):
        '''

        :return: changed task instruction
        '''
        string = ''
        obj_count = 0
        objs = list()

        was_delay = False
        for i, frame in enumerate(self.frame_info):
            delay_add = True
            for obj in self.frame_info.objset.dict[i]:
                obj_count += 1
                string += f'observe object {obj_count}, '
                objs.append(obj)
                delay_add, was_delay = False, False
            for d in frame.description:
                if 'ending' in d:
                    task_idx = int(re.search(r'\d+', d).group())
                    task_q = str(self.tasks[task_idx])
                    # align object numbers with lastk for each task
                    string += re.sub(pattern=f'last\d+ object',
                                     repl=ReplaceLastK(task_objset=self.task_objset[task_idx],
                                                       objs=objs), string=task_q)
                    delay_add, was_delay = False, False
            if delay_add and not was_delay:
                string += 'delay, '
                was_delay = True
        return string

    def get_target_value(self, examples):
        answers = list()
        for frame in self.frame_info:
            ending = False
            for d in frame.description:
                if 'ending' in d:
                    task_idx = int(re.search(r'\d+', d).group())
                    answers.append(examples[task_idx]['answers'][0])
                    ending = True
            if not ending:
                answers.append('null')
        return answers

    def get_examples(self):
        """
        get tasks examples
        :return: tuple of list of dictionaries containing information about the requested tasks
        and compo task
        """
        examples = list()
        for i, task in enumerate(self.tasks):
            examples.append({
                'family': str(task.__class__.__name__),
                # saving an epoch explicitly is needed because
                # there might be no objects in the last epoch.
                'epochs': int(task.n_frames),
                'question': str(task),
                'objects': [o.dump() for o in self.task_objset[i]],
                'answers': [const.get_target_value(t) for t in task.get_target(self.task_objset[i])],
                'first_shareable': int(task.first_shareable),
            })
        compo = {
            'epochs': int(len(self.frame_info)),
            'objects': [o.dump() for o in self.frame_info.objset],
            'instruction': str(self),
            'answers': self.get_target_value(examples)
        }
        return examples, compo

    @property
    def n_epochs(self):
        return len(self.frame_info)

    def get_changed_task_objset(self, changed_task):
        """

        :param changed_task:
        :return:
        """

        assert isinstance(changed_task, tg.TemporalTask)
        objset = changed_task.generate_objset(changed_task.n_distractors, changed_task.avg_mem)

        obj: sg.Object
        for obj in objset:
            for frame in self.frame_info:
                frame_obj: sg.Object
                for frame_obj in frame.objs:
                    if frame_obj.compare_attrs(obj):
                        obj.loc = frame_obj.loc
        return objset

    def reuse_merge(self, new_task_info):
        new_task = new_task_info.tasks[0]
        new_task_copy: tg.TemporalTask = new_task.copy()

        new_task_idx = len(self.tasks)

        start = self.frame_info.get_start_frame(new_task_info, relative_tasks={new_task_idx})
        objset = sg.ObjectSet(n_epoch=new_task_copy.n_frames, n_max_backtrack=(int(new_task_copy.avg_mem) * 3))
        changed = False
        # change the necessary selects first
        for i, (old_frame, new_frame) in enumerate(zip(self.frame_info[start:], new_task_info.frame_info)):
            lastk = 'last%d' % (len(new_task_info.frame_info) - i - 1)
            # if there are no objects in the one of the frames, then freely merge
            if old_frame.objs and new_frame.objs:
                filter_objs = new_task.reinit(new_task_copy, old_frame.objs, lastk)
                if filter_objs:
                    changed = True
                    for obj in filter_objs:
                        objset.add(obj=obj.copy(), epoch_now=new_task_copy.n_frames - 1, merge_idx=i)
                else:
                    raise RuntimeError('Unable to reuse')

        # get new objset for new task based on updated operators and merge

        if changed:
            # guess objset based on existing objects
            new_objset = new_task_copy.guess_objset(objset, new_task_copy.n_frames - 1)
            updated_fi = FrameInfo(new_task_copy, new_objset)
            FrameInfo.update_relative_tasks(updated_fi, relative_tasks={new_task_idx})
            for i, (old_frame, new_frame) in enumerate(zip(self.frame_info[start:], updated_fi)):
                old_frame.compatible_merge(new_frame)

            self.changed.append((new_task_idx, new_task, new_task_copy))
            self.task_objset[new_task_idx] = new_objset
        else:
            for i, (old_frame, new_frame) in enumerate(zip(self.frame_info[start:], new_task_info.frame_info)):
                old_frame.compatible_merge(new_frame)
            self.task_objset[new_task_idx] = new_task_info.task_objset[0]
        self.tasks.append(new_task)

    def merge(self, new_task_info, reuse=1):
        """
        combine new task to the existing composite task
        :param reuse: probability of reusing visual stimuli from previous composite task
        :param new_task_info: TaskInfoCombo object
        """
        assert isinstance(new_task_info, TaskInfoCompo)
        if len(new_task_info.tasks) > 1:
            raise NotImplementedError('Currently cannot support adding new composite tasks')

        if reuse is None:
            reuse = 0.5
        elif reuse == 1:
            return self.reuse_merge(new_task_info)

        new_task = new_task_info.tasks[0]
        new_task_copy: tg.TemporalTask = new_task.copy()
        new_task_idx = len(self.tasks)

        start = self.frame_info.get_start_frame(new_task_info, relative_tasks={new_task_idx})

        changed = False
        # TODO: fix action space
        for i, (old_frame, new_frame) in enumerate(zip(self.frame_info[start:], new_task_info.frame_info)):
            lastk = 'last%d' % (len(new_task_info.frame_info) - i - 1)

            # if there are no objects in the one of the frames, then freely merge
            if not old_frame.objs or not new_frame.objs:
                old_frame.compatible_merge(new_frame)  # always update frame descriptions
            else:
                # reuse stimuli with probability reuse
                if np.random.random() < reuse:
                    # check how many selects and if there are enough objects for the selects
                    if new_task.reinit(old_frame.objs, lastk, hard_update=False) and \
                            new_task_copy.reinit(old_frame.objs, lastk, hard_update=True):
                        changed = True
                        # select: object: view_angle: cateogry:

                        # what was the cateogry 1, object 2, viewangle 3 object
                    else:
                        old_frame.compatible_merge(new_frame)
                else:
                    old_frame.compatible_merge(new_frame)
        if changed:
            self.changed.append((new_task_idx, new_task, new_task_copy))
            self.task_objset[new_task_idx] = self.get_changed_task_objset(new_task_copy)
        else:
            self.task_objset[new_task_idx] = new_task_info.task_objset[0]
        self.tasks.append(new_task)


class FrameInfo(object):
    def __init__(self, task, objset):
        """
        used for combining multiple temporal tasks, initialize with 1 task,
        stores each frame object in frame_list
        :param task: stim_generator.Task object
        :param objset: objset related to the task
        """
        assert isinstance(objset, sg.ObjectSet)
        assert isinstance(task, tg.TemporalTask)

        if task.n_frames != objset.n_epoch:
            raise ValueError('Task epoch does not equal objset epoch')

        n_epochs = task.n_frames
        # TODO: decide if task_question needs to be kept
        task_question = str(task)
        task_answers = [const.get_target_value(t) for t in task.get_target(objset)]
        first_shareable = task.first_shareable

        relative_tasks = {0}
        self.objset = objset
        self.frame_list = list()
        self.n_epochs = n_epochs

        for i in range(n_epochs):
            description = list()
            if i == self.n_epochs - 1:
                description.append("ending of task %d" % 0)
            else:
                if i == 0:
                    description.append("start of task %d" % 0)
            self.frame_list.append(self.Frame(fi=self,
                                              idx=i,
                                              relative_tasks=relative_tasks,
                                              description=description))

        if objset:
            # iterate all objects in objset and add to each frame
            for obj in objset:
                for epoch in range(obj.epoch[0], obj.epoch[1]):
                    self.frame_list[epoch].objs.append(obj.copy())

        self.first_shareable = first_shareable

        self.last_task = 0
        self.last_task_end = len(self.frame_list) - 1
        self.last_task_start = 0

    def __len__(self):
        return len(self.frame_list)

    def __iter__(self):
        return self.frame_list.__iter__()

    def __getitem__(self, item):
        return self.frame_list.__getitem__(item)

    def dump(self):
        info = {i: dict() for i, frame in enumerate(self.frame_list)}
        for i, frame in enumerate(self.frame_list):
            info[i]['description'] = frame.description
            info[i]['relative_tasks'] = list(frame.relative_tasks)
            info[i]['relative_task_epoch_idx'] = frame.relative_task_epoch_idx
        return info

    def add_new_frames(self, i, relative_tasks):
        """
        add new empty frames and update objset and p
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
        # update the relative_task info for each frame in the new task (from 0 to new task idx)
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

    def get_start_frame(self, new_task_info, relative_tasks):
        """
        randomly sample a starting frame to start merging
        add new frames if needed
        check length of both, then starting first based on first_shareable
        if start at the same frame, but new task ends earlier,
        then start later, otherwise, new task can end earlier than existing task
        overall, task order is predetermined such that new task appears or finishes after the existing task
        avoid overlapping response frames
        :param relative_tasks: set of task indices used by the new task, relative to the
        TaskInfoCompo
        :param new_task_info: TaskInfoCompo object of the new task
        :return: index of frame from existing frame_info
        """
        assert isinstance(new_task_info, TaskInfoCompo)
        assert len(new_task_info.tasks) == 1

        first_shareable = self.first_shareable
        shareable_frames = self.frame_list[first_shareable:]
        new_task_len = new_task_info.n_epochs
        new_first_shareable = new_task_info.tasks[0].first_shareable

        self.update_relative_tasks(new_task_info.frame_info, relative_tasks)

        if len(shareable_frames) == 0:
            # queue, add frames and start merging
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

    class Frame(object):
        """ frame object within frame_info list"""

        def __init__(self, fi, idx, relative_tasks, description=None, objs=None):
            assert isinstance(fi, FrameInfo)
            assert isinstance(relative_tasks, set)

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

            self.relative_tasks = self.relative_tasks | new_frame.relative_tasks
            self.description = self.description + new_frame.description

            for new_obj in new_frame.objs:
                self.fi.objset.add(new_obj.copy(), len(self.fi) - 1, merge_idx=self.idx)
            for epoch, obj_list in self.fi.objset.dict.items():
                if epoch == self.idx:
                    self.objs = obj_list.copy()

            # update the dictionary for relative_task_epoch
            temp = self.relative_task_epoch_idx.copy()
            temp.update(new_frame.relative_task_epoch_idx)
            self.relative_task_epoch_idx = temp

        def __str__(self):
            return 'frame: ' + str(self.idx) + ', relative tasks: ' + \
                   ','.join([str(i) for i in self.relative_tasks]) \
                   + ' objects: ' + ','.join([str(o) for o in self.objs])
