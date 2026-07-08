"""
Focused unit and invariant tests for the composition and task-generation internals.

Unlike test_env.py (dataset-heavy, end-to-end smoke tests), these tests pin down
the contracts of individual components: frame placement, objset copying, frame
markers, merge invariants, instruction compilation, and graph building.
All tests are seeded and runnable from any working directory.
"""
import os
import random
import unittest

import numpy as np

from iwisdm import make
from iwisdm.envs.shapenet import task_generator as tg
from iwisdm.envs.shapenet import stim_generator as sg
from iwisdm.envs.shapenet import info_generator as ig
from iwisdm.envs.shapenet import auto_task_gen as atg
from iwisdm.envs.shapenet.info_generator import Frame, FrameInfo, TaskInfoCompo, InstructionCompiler
from iwisdm.envs.shapenet.registration import SNEnvSpec, DEFAULT_CONFIG
from iwisdm.envs.shapenet.task_bank import task_family_dict
from iwisdm.utils.helper import get_k

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
DATASET_FP = os.path.join(REPO_ROOT, 'data', 'shapenet_handpicked')

_ENV = None


def get_env():
    # one env per process: dataset loading is the slow part
    global _ENV
    if _ENV is None:
        _ENV = make(env_id='ShapeNet', dataset_fp=DATASET_FP)
    return _ENV


def seeded(seed=0):
    # the library samples from the global rngs during task instantiation
    random.seed(seed)
    np.random.seed(seed)


def make_task(n_frames=2):
    """a minimal deterministic TemporalTask: compare categories of the first/last frames"""
    if n_frames == 1:
        op = tg.GetCategory(tg.Select(when='last0'))
    else:
        op = tg.IsSame(
            tg.GetCategory(tg.Select(when=f'last{n_frames - 1}')),
            tg.GetCategory(tg.Select(when='last0')),
        )
    return tg.TemporalTask(operator=op, n_frames=n_frames, first_shareable=0)


@unittest.skipUnless(os.path.isdir(DATASET_FP), f'stimuli dataset not found at {DATASET_FP}')
class EnvTestCase(unittest.TestCase):
    """base class for tests that need stimuli data"""

    @classmethod
    def setUpClass(cls):
        cls.env = get_env()

    def setUp(self):
        self.env.reset_env()
        seeded(0)


class TestFindStartFrame(unittest.TestCase):
    """FrameInfo.find_start_frame is pure placement logic over plain ints"""

    def test_queue_case(self):
        # no shareable frames: task is queued after the timeline, which grows by its length
        self.assertEqual(FrameInfo.find_start_frame(
            first_shareable=3, n_frames=3, new_task_len=4, last_task_start=0, last_task_end=2),
            (3, 7))

    def test_exact_fit_shifts_right(self):
        # an exact fit would align response frames with the timeline end; must shift
        start, n = FrameInfo.find_start_frame(
            first_shareable=1, n_frames=4, new_task_len=3, last_task_start=0, last_task_end=3)
        self.assertEqual((start, n), (2, 5))

    def test_response_frame_collision_shifts_right(self):
        # start=0 would place the new response frame on last_task_end=3
        start, n = FrameInfo.find_start_frame(
            first_shareable=0, n_frames=5, new_task_len=4, last_task_start=0, last_task_end=3)
        self.assertNotEqual(start + 4 - 1, 3)
        self.assertFalse(start == 0 and start + 3 <= 3)

    def test_no_conflict_starts_at_first_shareable(self):
        start, n = FrameInfo.find_start_frame(
            first_shareable=1, n_frames=6, new_task_len=3, last_task_start=4, last_task_end=5)
        self.assertEqual(start, 1)
        self.assertEqual(n, 6)

    def test_invariants_hold_over_grid(self):
        # over a grid of valid inputs, placement never violates the ordering rules
        for fs in range(0, 5):
            for n0 in range(max(fs, 1), fs + 5):
                for L in range(1, 5):
                    for last_end in range(0, n0):
                        for last_start in range(0, last_end + 1):
                            start, n = FrameInfo.find_start_frame(fs, n0, L, last_start, last_end)
                            ctx = f'fs={fs} n0={n0} L={L} ls={last_start} le={last_end}'
                            self.assertGreaterEqual(start, fs, ctx)
                            self.assertGreaterEqual(n, n0, ctx)
                            # the timeline covers the new task
                            self.assertGreaterEqual(n, start + L, ctx)
                            # response frames never coincide
                            self.assertNotEqual(start + L - 1, last_end, ctx)
                            # if both tasks start together, the new one ends later
                            self.assertFalse(start == last_start and start + L - 1 <= last_end, ctx)


class TestObjectSetCopy(EnvTestCase):
    def _objset(self):
        task = make_task(n_frames=3)
        return task.generate_objset()

    def test_copy_preserves_order_and_content(self):
        objset = self._objset()
        copy = objset.copy()
        self.assertIsInstance(copy.set, list)  # regression: copy used to build a python set
        self.assertEqual([str(o) for o in objset.set], [str(o) for o in copy.set])
        self.assertEqual(objset.end_epoch, copy.end_epoch)
        self.assertEqual(sorted(objset.dict.keys()), sorted(copy.dict.keys()))
        for epoch in objset.dict:
            self.assertEqual([str(o) for o in objset.dict[epoch]],
                             [str(o) for o in copy.dict[epoch]])

    def test_copy_shares_instances_between_set_and_dict(self):
        # regression: dict entries used to be copied separately from set entries
        copy = self._objset().copy()
        set_ids = {id(o) for o in copy.set}
        for objs in copy.dict.values():
            for o in objs:
                self.assertIn(id(o), set_ids)

    def test_copy_is_independent(self):
        objset = self._objset()
        copy = objset.copy()
        for obj in objset.set:
            obj.epoch = [99, 100]
        for obj in copy.set:
            self.assertNotEqual(obj.epoch, [99, 100])

    def test_copy_supports_add(self):
        # regression: adding to a copy used to crash on set.insert
        copy = self._objset().copy()
        added = copy.add(sg.Object(when='last0'), epoch_now=copy.n_epoch - 1)
        self.assertIsNotNone(added)


class TestFrameMarkers(EnvTestCase):
    def test_single_frame_task_markers(self):
        task = make_task(n_frames=1)
        fi = FrameInfo(task, task.generate_objset())
        self.assertEqual(len(fi), 1)
        self.assertEqual(fi[0].task_ends, {0})
        self.assertEqual(fi[0].task_starts, set())

    def test_multi_frame_task_markers(self):
        task = make_task(n_frames=3)
        fi = FrameInfo(task, task.generate_objset())
        self.assertEqual(fi[0].task_starts, {0})
        self.assertEqual(fi[0].task_ends, set())
        self.assertEqual(fi[2].task_ends, {0})
        for frame in fi:
            self.assertEqual(frame.relative_tasks, {0})

    def test_description_property_format(self):
        frame = Frame(idx=1, relative_tasks={0, 2}, task_starts={2}, task_ends={0})
        self.assertEqual(frame.description, ['start of task 2', 'ending of task 0'])
        distractor = Frame(idx=0, relative_tasks={-1, 0}, distractor_attrs=['category'])
        self.assertEqual(distractor.description, ['distractor with different category'])
        plain_distractor = Frame(idx=0, relative_tasks={-1, 0})
        self.assertEqual(plain_distractor.description, ['distractor'])

    def test_update_relative_tasks_renames(self):
        task = make_task(n_frames=2)
        fi = FrameInfo(task, task.generate_objset())
        FrameInfo.update_relative_tasks(fi, {3})
        self.assertEqual(fi[0].relative_tasks, {3})
        self.assertEqual(fi[0].task_starts, {3})
        self.assertEqual(fi[1].task_ends, {3})
        # a second rename must fail: the frames no longer belong to task 0
        with self.assertRaises(RuntimeError):
            FrameInfo.update_relative_tasks(fi, {4})

    def test_merge_frame_unions_markers(self):
        task = make_task(n_frames=2)
        fi = FrameInfo(task, task.generate_objset())
        incoming = Frame(idx=0, relative_tasks={1}, task_starts={1}, distractor_attrs=['location'])
        fi.merge_frame(0, incoming)
        self.assertEqual(fi[0].relative_tasks, {0, 1})
        self.assertEqual(fi[0].task_starts, {0, 1})
        self.assertEqual(fi[0].distractor_attrs, ['location'])

    def test_frame_copy_is_independent(self):
        frame = Frame(idx=0, relative_tasks={0}, task_starts={0})
        copy = frame.copy()
        copy.relative_tasks.add(5)
        copy.task_starts.add(5)
        self.assertEqual(frame.relative_tasks, {0})
        self.assertEqual(frame.task_starts, {0})


class TestComposition(EnvTestCase):
    def _merged(self, n_tasks=3, num_merge=2, seed=0):
        seeded(seed)
        tasks = [t[1][1] for t in self.env.generate_tasks(n_tasks)]
        return self.env.merge_tasks(tasks=tasks, num_merge=num_merge)

    def test_merge_does_not_mutate_argument(self):
        seeded(0)
        tasks = [t[1][1] for t in self.env.generate_tasks(2)]
        infos = self.env.init_compositional_tasks([t.copy() for t in tasks])
        base, arg = infos
        arg_task_str = str(arg.tasks[0])
        arg_frames = [str(f) for f in arg.frame_info]
        arg_marks = [(set(f.task_starts), set(f.task_ends), set(f.relative_tasks))
                     for f in arg.frame_info]
        base.merge(arg)
        self.assertEqual(str(arg.tasks[0]), arg_task_str)
        self.assertEqual([str(f) for f in arg.frame_info], arg_frames)
        self.assertEqual(
            [(set(f.task_starts), set(f.task_ends), set(f.relative_tasks)) for f in arg.frame_info],
            arg_marks)

    def test_composed_records_are_consistent(self):
        compo = self._merged(n_tasks=4, num_merge=3)
        n_tasks = len(compo.composed)
        self.assertEqual(n_tasks, 4)  # base + 3 merges
        for i, record in enumerate(compo.composed):
            end = record.start_frame + record.task.n_frames - 1
            self.assertLess(end, len(compo.frame_info),
                            f'task {i} ends outside the timeline')
            self.assertIn(i, compo.frame_info[end].task_ends,
                          f'task {i} response frame lacks its ending marker')
            # every frame in the task's span belongs to it
            for idx in range(record.start_frame, end + 1):
                self.assertIn(i, compo.frame_info[idx].relative_tasks)
        # each task ends exactly once across the timeline
        all_ends = [i for frame in compo.frame_info for i in frame.task_ends]
        self.assertEqual(sorted(all_ends), list(range(n_tasks)))

    def test_compat_properties(self):
        compo = self._merged()
        self.assertEqual(len(compo.tasks), len(compo.composed))
        self.assertEqual(set(compo.task_objset.keys()), set(range(len(compo.composed))))
        for record, task in zip(compo.composed, compo.tasks):
            self.assertIs(record.task, task)

    def test_answers_align_with_endings(self):
        compo = self._merged(n_tasks=3, num_merge=2)
        per_task_info, compo_info = compo.get_task_info_dict()
        answers = compo_info['answers']
        epochs = compo_info['epochs']
        n_tasks = len(per_task_info)
        self.assertEqual(epochs, len(compo.frame_info))
        # one answer per task ending, one null per non-ending frame
        self.assertEqual(sum(a != 'null' for a in answers), n_tasks)
        self.assertGreaterEqual(len(answers), epochs)
        self.assertLessEqual(len(answers), epochs + n_tasks - 1)

    def test_instruction_references_every_object(self):
        compo = self._merged(n_tasks=3, num_merge=2)
        _, compo_info = compo.get_task_info_dict()
        instruction = compo_info['instruction']
        for obj in compo_info['objects']:
            if not obj['obj']['is_distractor']:
                self.assertIn(f'object {obj["count"]}', instruction)

    def test_copy_is_deep_and_faithful(self):
        compo = self._merged()
        copy = compo.copy()
        self.assertEqual(len(copy.composed), len(compo.composed))
        self.assertIsNot(copy.frame_info, compo.frame_info)
        _, info_a = compo.get_task_info_dict()
        _, info_b = copy.get_task_info_dict()
        self.assertEqual(info_a['instruction'], info_b['instruction'])
        self.assertEqual(info_a['answers'], info_b['answers'])
        copy.frame_info[0].task_starts.add(99)
        copy.composed[0].objset.set[0].epoch = [98, 99]
        self.assertNotIn(99, compo.frame_info[0].task_starts)
        self.assertNotEqual(compo.composed[0].objset.set[0].epoch, [98, 99])

    def test_merge_tasks_does_not_mutate_pool(self):
        seeded(0)
        tasks = [t[1][1] for t in self.env.generate_tasks(3)]
        infos = self.env.init_compositional_tasks([t.copy() for t in tasks])
        pool = list(infos)
        self.env.merge_tasks(compositional_infos=infos, num_merge=2)
        self.assertEqual(infos, pool)

    def test_trial_generation_consistency(self):
        compo = self._merged(n_tasks=3, num_merge=2)
        imgs, per_task_info, compo_info = compo.generate_trial(
            self.env.env_spec.canvas_size,
            fixation_cue=True,
            cue_on_action=False,
            stim_data=self.env.stim_data,
        )
        self.assertEqual(len(imgs), compo_info['epochs'])
        self.assertEqual(len(per_task_info), len(compo.composed))

    def test_distractor_frame_markers(self):
        compo = self._merged(n_tasks=2, num_merge=1)
        compo.add_distractor_frame(2, self.env.stim_data)
        distractor_frames = [f for f in compo.frame_info if -1 in f.relative_tasks]
        self.assertTrue(distractor_frames, 'no distractor was added')
        for frame in distractor_frames:
            self.assertTrue(frame.distractor_attrs,
                            'frame distractors must record their differing attribute')
            self.assertEqual(len(frame.objs), 2)
        # instruction generation still works and mentions the differing attribute
        _, compo_info = compo.get_task_info_dict()
        self.assertTrue(any(f' with {a}: ' in compo_info['instruction']
                            or f' at {a}: ' in compo_info['instruction']
                            for f in distractor_frames for a in f.distractor_attrs))

    def test_distractor_time_markers(self):
        compo, empty_before = None, []
        for seed in range(8):  # find a composition with an empty (delay) frame to fill
            candidate = self._merged(n_tasks=2, num_merge=1, seed=seed)
            empty_before = [i for i, f in enumerate(candidate.frame_info) if not f.objs]
            if empty_before:
                compo = candidate
                break
        if compo is None:
            self.skipTest('no seed produced a composition with empty frames')
        compo.add_distractor_time(1, self.env.stim_data)
        filled = [i for i in empty_before if compo.frame_info[i].objs]
        self.assertEqual(len(filled), 1)
        frame = compo.frame_info[filled[0]]
        self.assertIn(-1, frame.relative_tasks)
        self.assertTrue(all(o.deletable for o in frame.objs))
        # time distractors have no differing-attribute marker
        self.assertEqual(frame.distractor_attrs, [])


class TestInstructionCompiler(unittest.TestCase):
    def test_object_add_shifts_counts(self):
        text = 'observe object 1, delay, observe object 2, category of object 1 equals category of object 2?'
        shifted = InstructionCompiler.object_add(text, 3)
        self.assertEqual(
            shifted,
            'observe object 4, delay, observe object 5, category of object 4 equals category of object 5?')

    def test_object_add_zero_is_identity(self):
        text = 'observe object 7, location of object 7?'
        self.assertEqual(InstructionCompiler.object_add(text, 0), text)


def _mini_spec(op_dict, root_ops, **overrides):
    config = {
        'op_dict': op_dict,
        'root_ops': root_ops,
        'boolean_ops': root_ops,
        'max_op': overrides.pop('max_op', 20),
        'max_depth': overrides.pop('max_depth', 10),
        'max_switch': 0,
        'switch_threshold': 0.0,
        'select_limit': False,
    }
    config.update(overrides)
    return SNEnvSpec(auto_gen_config=config)


class TestTaskGeneration(unittest.TestCase):
    def _default_gen(self, seed=0):
        return atg.SNTaskGenerator(SNEnvSpec(auto_gen_config=DEFAULT_CONFIG), rng=random.Random(seed))

    def test_same_seed_reproduces_graphs(self):
        def structures(seed):
            gen = self._default_gen(seed)
            out = []
            for _ in range(20):
                G, _, _ = gen.subtask_graph_generator(max_op=20, max_depth=10)
                out.append(sorted((u, v, G.nodes[u]['label'], G.nodes[v]['label'])
                                  for u, v in G.edges))
            return out

        self.assertEqual(structures(42), structures(42))
        self.assertNotEqual(structures(42), structures(7))

    def test_graph_structure_sanity(self):
        gen = self._default_gen()
        valid_labels = set(DEFAULT_CONFIG['op_dict'].keys()) | {'CONST'}
        for _ in range(50):
            G, root, op_count = gen.subtask_graph_generator(max_op=20, max_depth=10)
            # node ids are allocated contiguously from the root
            self.assertEqual(sorted(G.nodes), list(range(root, op_count + 1)))
            self.assertEqual(G.in_degree(root), 0)
            for node in G.nodes:
                self.assertIn(G.nodes[node]['label'], valid_labels)
                if node != root:
                    self.assertEqual(G.in_degree(node), 1)

    def test_select_limit_blocks_get_children(self):
        gen = self._default_gen()
        for _ in range(50):
            G, _, _ = gen.subtask_graph_generator(max_op=20, max_depth=10, select_limit=True)
            for u, v in G.edges:
                if G.nodes[u]['label'] == 'Select':
                    self.assertFalse(G.nodes[v]['label'].startswith('Get'),
                                     'select_limit=True must not grow Get ops below Selects')

    def test_select_sampling_does_not_mutate_config(self):
        # regression: sampling used to remove entries from op_dict['Select']['downstream']
        spec = SNEnvSpec(auto_gen_config=DEFAULT_CONFIG)
        gen = atg.SNTaskGenerator(spec, rng=random.Random(0))
        before = list(spec.auto_gen_config['op_dict']['Select']['downstream'])
        for _ in range(200):
            gen.sample_children_op(
                op_name='Select', op_count=1, max_op=20, cur_depth=1, max_depth=10,
                select_op=True, select_downstream=None)
        self.assertEqual(spec.auto_gen_config['op_dict']['Select']['downstream'], before)


class TestBudgetSampling(unittest.TestCase):
    """sample_children_helper prefers ops that fit the budgets with slack"""

    OP_DICT = {
        'A': {'n_downstream': 1, 'downstream': ['Cheap', 'Fits', 'Big'],
              'same_children_op': False, 'min_depth': 1, 'min_op': 1},
        'Cheap': {'n_downstream': 0, 'downstream': [], 'same_children_op': False,
                  'min_depth': 2, 'min_op': 1},
        'Fits': {'n_downstream': 0, 'downstream': [], 'same_children_op': False,
                 'min_depth': 2, 'min_op': 3},
        'Big': {'n_downstream': 0, 'downstream': [], 'same_children_op': False,
                'min_depth': 1, 'min_op': 9},
    }

    def _gen(self, seed=0):
        return atg.SNTaskGenerator(_mini_spec(self.OP_DICT, ['A']), rng=random.Random(seed))

    def _draws(self, gen, op_count, max_op, cur_depth=0, max_depth=10, n=100):
        return {gen.sample_children_helper('A', op_count, max_op, cur_depth, max_depth)
                for _ in range(n)}

    def test_prefers_positive_slack(self):
        # Cheap leaves slack, Fits is an exact fit, Big does not fit
        self.assertEqual(self._draws(self._gen(), op_count=1, max_op=4), {'Cheap'})

    def test_exact_fit_when_no_slack(self):
        # only Cheap fits, exactly
        self.assertEqual(self._draws(self._gen(), op_count=1, max_op=2), {'Cheap'})

    def test_fallback_when_nothing_fits(self):
        # no op fits the operator budget: fall back to the minimum-depth op (Big)
        self.assertEqual(self._draws(self._gen(), op_count=1, max_op=1), {'Big'})

    def test_depth_budget_filters(self):
        # at depth 9 of 10 only Big (min_depth 1) still fits
        self.assertEqual(
            self._draws(self._gen(), op_count=1, max_op=20, cur_depth=9, max_depth=10),
            {'Big'})

    def test_sample_dist_overrides_budgets(self):
        op_dict = {k: dict(v) for k, v in self.OP_DICT.items()}
        op_dict['A'] = dict(op_dict['A'], sample_dist=[0.0, 1.0, 0.0])
        gen = atg.SNTaskGenerator(_mini_spec(op_dict, ['A']), rng=random.Random(0))
        self.assertEqual(self._draws(gen, op_count=1, max_op=1), {'Fits'})


class TestConstSwap(unittest.TestCase):
    def _gen(self, compare_const_prob, seed=0):
        op_dict = {
            'IsSame': {'n_downstream': 2, 'downstream': ['GetLoc', 'GetCategory'],
                       'same_children_op': True, 'min_depth': 3, 'min_op': 7},
            'GetLoc': {'n_downstream': 1, 'downstream': ['Select'],
                       'same_children_op': False, 'min_depth': 2, 'min_op': 2},
            'GetCategory': {'n_downstream': 1, 'downstream': ['Select'],
                            'same_children_op': False, 'min_depth': 2, 'min_op': 2},
            'GetObject': {'n_downstream': 1, 'downstream': ['Select'],
                          'same_children_op': False, 'min_depth': 2, 'min_op': 2},
            'Select': {'n_downstream': 4, 'downstream': [],
                       'same_children_op': False, 'min_depth': 1, 'min_op': 1},
        }
        spec = _mini_spec(
            op_dict, ['IsSame'],
            compare_const_prob=compare_const_prob,
            const_parent_ops=['IsSame'],
            indexable_get_ops=['GetLoc', 'GetCategory'],
        )
        return atg.SNTaskGenerator(spec, rng=random.Random(seed))

    def test_prob_one_swaps_exactly_one_indexable_child(self):
        gen = self._gen(compare_const_prob=1.0)
        for _ in range(50):
            children = gen.maybe_swap_in_const('IsSame', ['GetLoc', 'GetLoc'])
            self.assertEqual(children.count('CONST'), 1)
            self.assertIn(children[1 - children.index('CONST')], ['GetLoc'])

    def test_prob_one_forces_indexable_children(self):
        gen = self._gen(compare_const_prob=1.0)
        for _ in range(50):
            children = gen.maybe_swap_in_const('IsSame', ['GetObject', 'GetObject'])
            self.assertEqual(children.count('CONST'), 1)
            other = children[1 - children.index('CONST')]
            self.assertIn(other, ['GetLoc', 'GetCategory'])

    def test_prob_zero_is_identity(self):
        gen = self._gen(compare_const_prob=0.0)
        children = ['GetLoc', 'GetLoc']
        self.assertEqual(gen.maybe_swap_in_const('IsSame', list(children)), children)

    def test_non_const_parent_is_identity(self):
        gen = self._gen(compare_const_prob=1.0)
        children = ['GetLoc', 'GetLoc']
        self.assertEqual(gen.maybe_swap_in_const('GetLoc', list(children)), children)


class TestSampleWhen(unittest.TestCase):
    def test_no_delay_gives_contiguous_frames(self):
        spec = SNEnvSpec(max_delay=3, delay_prob=0.0)
        self.assertEqual(spec.sample_when(4), ['last0', 'last1', 'last2', 'last3'])

    def test_full_delay_is_bounded_by_max_delay(self):
        spec = SNEnvSpec(max_delay=3, delay_prob=1.0)
        # each step inserts a delay until MAX_DELAY is exhausted
        self.assertEqual(spec.sample_when(5), ['last0', 'last2', 'last4', 'last6', 'last7'])

    def test_whens_are_distinct_and_ordered(self):
        spec = SNEnvSpec(max_delay=4, delay_prob=0.5)
        random.seed(1)
        for n in range(1, 6):
            whens = spec.sample_when(n)
            ks = [get_k(w) for w in whens]
            self.assertEqual(len(whens), n)
            self.assertEqual(ks[0], 0)
            self.assertEqual(ks, sorted(ks))
            self.assertEqual(len(set(ks)), n, 'whens must be distinct')
            # total delays (skipped ks) never exceed MAX_DELAY
            self.assertLessEqual(max(ks) - (n - 1), spec.MAX_DELAY)


class TestGeneratedTasks(EnvTestCase):
    def test_generate_task_end_to_end(self):
        # building operators from graphs needs the env's class-level spec, hence EnvTestCase
        gen = atg.SNTaskGenerator(self.env.env_spec, rng=random.Random(0))
        for _ in range(10):
            (G, root, op_count), (op, task) = gen.generate_task()
            self.assertIsInstance(task, tg.TemporalTask)
            self.assertGreaterEqual(task.n_frames, 1)
            self.assertTrue(str(task))


class TestPremadeTasks(EnvTestCase):
    def test_premade_task_generates_trials(self):
        seeded(3)
        self.env.env_spec.delay_prob = 0.5
        # this dataset has a single view angle per object, so view-angle
        # comparisons are not instantiable; test every other family
        supported = [f for f in task_family_dict if 'ViewAngle' not in f]
        for family in supported:
            task = self.env.get_premade_task(task_family=[family])
            self.assertIsInstance(task, tg.TemporalTask, family)
            imgs, per_task_info, compo_info = self.env.generate_trials(tasks=[task])[0]
            self.assertEqual(len(imgs), task.n_frames, family)
            self.assertEqual(compo_info['epochs'], task.n_frames, family)

    def test_view_angle_comparison_raises_on_unsupported_dataset(self):
        # regression: comparing view angles on a dataset with one view angle
        # per object used to loop forever; it must raise instead
        seeded(0)
        task = self.env.get_premade_task(task_family=['CompareViewAngle'])
        with self.assertRaises(ValueError):
            self.env.generate_trials(tasks=[task])


if __name__ == '__main__':
    unittest.main()
