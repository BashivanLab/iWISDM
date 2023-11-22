import torch

# generate tasks with specified task name variable
from cognitive import info_generator as ig
from cognitive import stim_generator as sg

frame_info = ig.FrameInfo(self.task, self.task.generate_objset())
compo_info = ig.TaskInfoCompo(self.task, frame_info)
objset = compo_info.frame_info.objset

images = []
for epoch, frame in zip(sg.render(objset, self.img_size, train=self.train), compo_info.frame_info):
    if self.fixation_cue:
        if not any('ending' in description for description in frame.description):
            sg.add_fixation_cue(epoch)
        img = Image.fromarray(epoch, 'RGB')
        images.append(img)
    _, data, _ = compo_info.get_examples()