import unittest

from cognitive import info_generator as ig
from cognitive import stim_generator as sg
from cognitive import task_bank as tb
from cognitive import task_generator as tg

families = list(tb.task_family_dict.keys())


class InfoGeneratorTest(unittest.TestCase):
    def testMerge(self):
        categories = sg.sample_category(3)
        while len(shapes) != len(set(shapes)):
            shapes = sg.sample_shape(3)
        colors = sg.sample_color(3)
        while len(colors) != len(set(colors)):
            colors = sg.sample_color(3)
        whens = sg.sample_when(2)
        while len(whens) != len(set(whens)):
            whens = sg.sample_when(2)

        n_frames = 3
        op1 = tg.Select(color=colors[2], shape=shapes[2], when=f'last{n_frames - 1}')
        op2 = tg.Select(color=colors[0], shape=shapes[0], when=f'last{n_frames - 2}')
        new_task1 = tg.TemporalTask(tg.IsSame(tg.GetShape(op1), tg.GetShape(op2)), n_frames, 1)
        new_task1_objset = new_task1.generate_objset()
        fi1 = ig.FrameInfo(new_task1, new_task1_objset)
        compo_info = ig.TaskInfoCompo(new_task1, fi1)

        self.assertTrue(len(compo_info.frame_info.objset) == 2)

        op3 = tg.Select(color=colors[1], shape=shapes[1], when=f'last{n_frames - 1}')
        new_task2 = tg.TemporalTask(tg.GetShape(op3), n_frames)
        new_task2_objset = new_task2.generate_objset()
        fi2 = ig.FrameInfo(new_task2, new_task2_objset)
        compo_info_2 = ig.TaskInfoCompo(new_task2, fi2)

        self.assertTrue(len(compo_info_2.frame_info.objset) == 1)
        self.assertEqual(compo_info_2.frame_info[0].objs[0].color, colors[1])

        compo_info.merge(compo_info_2, reuse=1)
        self.assertEqual(2, len(compo_info.frame_info.objset))
        self.assertEqual(op3.shape, shapes[0])

        op1 = tg.Select(color=colors[2], when=f'last{n_frames - 1}')
        op2 = tg.Select(color=colors[0], when=f'last{n_frames - 2}')
        new_task1 = tg.TemporalTask(tg.IsSame(tg.GetShape(op1), tg.GetShape(op2)), n_frames, 1)
        new_task1_objset = new_task1.generate_objset()
        fi1 = ig.FrameInfo(new_task1, new_task1_objset)
        compo_info = ig.TaskInfoCompo(new_task1, fi1)

        op3 = tg.Select(when=f'last{n_frames - 1}')
        new_task2 = tg.TemporalTask(tg.GetShape(op3), n_frames)
        new_task2_objset = new_task2.generate_objset()
        fi2 = ig.FrameInfo(new_task2, new_task2_objset)
        compo_info_2 = ig.TaskInfoCompo(new_task2, fi2)

        compo_info.merge(compo_info_2, reuse=1)
        self.assertTrue(all(obj1.compare_attrs(obj2, ['color', 'shape', 'loc'])
                            for obj1, obj2 in zip(compo_info.task_objset[1].set, compo_info.frame_info[1].objs)))

    def testChangeTaskObjset(self):
        shapes = sg.sample_shape(3)
        while len(shapes) != len(set(shapes)):
            shapes = sg.sample_shape(3)
        colors = sg.sample_color(3)
        while len(colors) != len(set(colors)):
            colors = sg.sample_color(3)
        whens = sg.sample_when(2)
        while len(whens) != len(set(whens)):
            whens = sg.sample_when(2)

        task = tb.random_task(families)
        while task.n_frames < 3:
            task = tb.random_task(families)
        _ = task.generate_objset()

        loc1 = sg.Loc((0.111, 0.111))
        loc2 = sg.Loc((0.001, 0.001))
        object1 = sg.Object(attrs=[colors[2], shapes[2], loc1], when=f'last{task.n_frames - 1}')
        object2 = sg.Object(attrs=[colors[0], shapes[0], loc2], when=f'last{task.n_frames - 2}')
        objset = sg.ObjectSet(task.n_frames, int(task.avg_mem * 3))
        objset.add(object1, task.n_frames - 1)
        objset.add(object2, task.n_frames - 1)

        fi = ig.FrameInfo(task, objset)
        compo_info = ig.TaskInfoCompo(task, fi)

        op1 = tg.Select(color=colors[2], shape=shapes[2], when=f'last{task.n_frames - 1}')
        op2 = tg.Select(color=colors[0], shape=shapes[0], when=f'last{task.n_frames - 2}')
        new_task1 = tg.TemporalTask(tg.IsSame(tg.GetShape(op1), tg.GetShape(op2)))
        new_task1.n_frames = task.n_frames
        ori_objset = new_task1.generate_objset()
        changed_objset = compo_info.get_changed_task_objset(new_task1)

        changed_obj: sg.Object
        ori_obj: sg.Object
        for changed_obj, ori_obj in zip(changed_objset, ori_objset):
            self.assertEqual(changed_obj.shape, ori_obj.shape)
            self.assertEqual(changed_obj.color, ori_obj.color)
            self.assertIn(changed_obj.loc, [loc1, loc2])


if __name__ == '__main__':
    unittest.main()
