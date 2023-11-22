import unittest

from cognitive import info_generator as ig
from cognitive import stim_generator as sg
from cognitive import task_bank as tb
from cognitive import task_generator as tg
from cognitive import constants as const

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

    def testString(self):
        const.DATA = const.Data('../data/min_shapenet_easy_angle')
        categories = sg.sample_category(4)
        objects = [sg.sample_object(k=1, category=cat)[0] for cat in categories]
        view_angles = [sg.sample_view_angle(k=1, obj=obj)[0] for obj in objects]

        op1 = tg.Select(category=categories[0], when=f'last{3 - 1}')
        op2 = tg.Select(category=categories[1], when=f'last{3 - 2}')
        new_task1 = tg.TemporalTask(tg.IsSame(tg.GetCategory(op1), tg.GetCategory(op2)), 3)
        op3 = tg.Select(category=categories[2], when=f'last{3}')
        op4 = tg.Select(category=categories[3], when=f'last{2}')
        new_task2 = tg.TemporalTask(tg.IsSame(tg.GetCategory(op3), tg.GetCategory(op4)), 4)

        new_task1_objset = new_task1.generate_objset()
        fi1 = ig.FrameInfo(new_task1, new_task1_objset)
        compo_info = ig.TaskInfoCompo(new_task1, fi1)

        new_task2_objset = new_task2.generate_objset()
        fi2 = ig.FrameInfo(new_task2, new_task2_objset)
        compo_info2 = ig.TaskInfoCompo(new_task2, fi2)
        compo_info.merge(compo_info2)
        print(new_task1.n_frames, new_task2.n_frames, new_task1.first_shareable)
        print(compo_info.n_epochs)
        print(compo_info.get_examples()[1]['objects'])
        print(compo_info)
        print(new_task2_objset)

    def testReuseMerge(self):
        categories = sg.sample_category(4)
        objects = [sg.sample_object(k=1, category=cat)[0] for cat in categories]
        view_angles = [sg.sample_view_angle(k=1, obj=obj)[0] for obj in objects]

        op1 = tg.Select(category=categories[0], when=f'last{3 - 1}')
        op2 = tg.Select(category=categories[1], when=f'last{3 - 2}')
        new_task1 = tg.TemporalTask(tg.IsSame(tg.GetCategory(op1), tg.GetCategory(op2)), 3)

        op3 = tg.Select(category=categories[2], when=f'last{3}')
        op4 = tg.Select(category=categories[3], when=f'last{2}')
        new_task2 = tg.TemporalTask(tg.IsSame(tg.GetCategory(op3), tg.GetCategory(op4)), 4)

        new_task1_objset = new_task1.generate_objset()
        fi1 = ig.FrameInfo(new_task1, new_task1_objset)
        compo_info = ig.TaskInfoCompo(new_task1, fi1)
        new_task2_objset = new_task2.generate_objset()
        fi2 = ig.FrameInfo(new_task2, new_task2_objset)
        compo_info2 = ig.TaskInfoCompo(new_task2, fi2)
        compo_info.merge(compo_info2, reuse=1)

        objlist: list[sg.Object]
        for epoch, objlist in compo_info.frame_info.objset.dict.items():
            if epoch == 0:
                self.assertEqual(objlist[0].category, categories[0])
            elif epoch == 1:
                self.assertEqual(objlist[0].category, categories[1])
            else:
                self.assertTrue(len(objlist) == 0)

        op1 = tg.Select(category=categories[0], when=f'last{3 - 1}')
        op2 = tg.Select(category=categories[1], when=f'last{3 - 2}')
        new_task1 = tg.TemporalTask(tg.IsSame(tg.GetCategory(op1), tg.GetCategory(op2)), 3, 0)

        op3 = tg.Select(category=categories[2], when=f'last{3}')
        op4 = tg.Select(category=categories[3], when=f'last{2}')
        new_task2 = tg.TemporalTask(tg.IsSame(tg.GetCategory(op3), tg.GetCategory(op4)), 4, 0)

        new_task1_objset = new_task1.generate_objset()
        fi1 = ig.FrameInfo(new_task1, new_task1_objset)
        compo_info = ig.TaskInfoCompo(new_task1, fi1)
        new_task2_objset = new_task2.generate_objset()
        fi2 = ig.FrameInfo(new_task2, new_task2_objset)
        compo_info2 = ig.TaskInfoCompo(new_task2, fi2)
        compo_info.merge(compo_info2, reuse=1)

        op1 = tg.Select(when=f'last1')
        op2 = tg.Select(when=f'last3')
        new_task1 = tg.TemporalTask(tg.IsSame(tg.GetLoc(op1), tg.GetLoc(op2)), 4, first_shareable=4)

        op3 = tg.Select(when=f'last0')
        op4 = tg.Select(when=f'last4')
        new_task2 = tg.TemporalTask(tg.IsSame(tg.GetLoc(op3), tg.GetLoc(op4)), 5, 0)

        new_task1_objset = new_task1.generate_objset()
        fi1 = ig.FrameInfo(new_task1, new_task1_objset)
        compo_info = ig.TaskInfoCompo(new_task1, fi1)
        new_task2_objset = new_task2.generate_objset()
        fi2 = ig.FrameInfo(new_task2, new_task2_objset)
        compo_info2 = ig.TaskInfoCompo(new_task2, fi2)
        compo_info.merge(compo_info2, reuse=1)
        print(str(compo_info))

    def testTemporalSwitch(self):
        const.DATA = const.Data()
        categories = sg.sample_category(4)
        objects = [sg.sample_object(k=1, category=cat)[0] for cat in categories]
        view_angles = [sg.sample_view_angle(k=1, obj=obj)[0] for obj in objects]

        op1 = tg.Select(category=categories[0], when='last2')
        op2 = tg.Select(category=categories[1], when=f'last1')
        new_task1 = tg.TemporalTask(tg.IsSame(tg.GetCategory(op1), tg.GetCategory(op2)), 3)

        op3 = tg.Select(category=categories[2], when='last3')
        op4 = tg.Select(category=categories[3], when='last2')
        new_task2 = tg.TemporalTask(tg.IsSame(tg.GetCategory(op3), tg.GetCategory(op4)), 4)
        op3 = tg.Select(object=objects[0], when='last3')
        op4 = tg.Select(object=objects[1], when='last2')
        new_task3 = tg.TemporalTask(tg.IsSame(tg.GetObject(op3), tg.GetObject(op4)), 4)

        new_task1_objset = new_task1.generate_objset()
        fi1 = ig.FrameInfo(new_task1, new_task1_objset)
        compo_info = ig.TaskInfoCompo(new_task1, fi1)
        new_task2_objset = new_task2.generate_objset()
        fi2 = ig.FrameInfo(new_task2, new_task2_objset)
        compo_info2 = ig.TaskInfoCompo(new_task2, fi2)
        new_task3_objset = new_task3.generate_objset()
        fi3 = ig.FrameInfo(new_task3, new_task3_objset)
        compo_info3 = ig.TaskInfoCompo(new_task3, fi3)

        compo_info.temporal_switch(compo_info2, compo_info3)
        print(compo_info.get_examples()[1]['instruction'])

    def testObjectAddOne(self):
        t = 'observe object 1, observe object 2, category of object 1 equal category of object 2 ?observe object 3, observe object 4, delay, object of object 3 equal object of object 4 ?'
        print(ig.object_add(t, 2))


if __name__ == '__main__':
    unittest.main()
