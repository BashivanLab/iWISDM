from cognitive.task_generator import TemporalTask
from cognitive import stim_generator as sg
from cognitive import task_generator as tg
from cognitive import constants as const
from cognitive import info_generator as ig
import json
import torch
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import Dataset

class CompareCategoryTemporal(TemporalTask):
    """Compare location between two objects."""

    def __init__(self, whens):
        """
            :param whens: the list of two frame names to compare stimuli location between
        """

        # Initialize Class with parent class
        super(CompareCategoryTemporal, self).__init__(whens=whens)

        # Select the specified frames
        objs1 = tg.Select(when=self.whens[0])
        objs2 = tg.Select(when=self.whens[1])

        # Get the locations of stimuli within each frame
        a1 = tg.GetCategory(objs1)
        a2 = tg.GetCategory(objs2)

        # Set operator to check if they're the same location
        self._operator = tg.IsSame(a1, a2)

        # 
        self.n_frames = const.compare_when([self.whens[0], self.whens[1]]) + 1


# stim_dir = '/home/xiaoxuan/projects/multfs_triple/Lucas_scripts/data/new_shapenet_train'
# const.DATA = const.Data(stim_dir)

# whens = ['last0', 'last2']
# comp_loc_task = CompareCategoryTemporal(whens)
# frame_info = ig.FrameInfo(comp_loc_task, comp_loc_task.generate_objset())
# compo_info = ig.TaskInfoCompo(comp_loc_task,frame_info)
# img, action = compo_info.generate_trial()
# print(img.shape) # (224,224,4)

class DMSCtg_TaskDataset(Dataset):
    def __init__(self, stim_dir, whens = ['last0', 'last2'], phase = "train",dataset_size = 2560):
        self.phase = phase
        self.stim_dir = stim_dir
        self.dataset_size = dataset_size
        const.DATA = const.Data(self.stim_dir)
        self.comp_loc_task = CompareCategoryTemporal(whens)
        self.frame_info = ig.FrameInfo(self.comp_loc_task, self.comp_loc_task.generate_objset())
        self.compo_info = ig.TaskInfoCompo(self.comp_loc_task,self.frame_info)
        # preprocessing steps for pretrained ResNet models
        self.transform = transforms.Compose([
                            transforms.Resize(224),
                            # transforms.CenterCrop(224), # todo: to delete for shapenet task; why?
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        const.DATA = const.Data(self.stim_dir, phase = self.phase)
        imgs, action = self.compo_info.generate_trial()
        for i, img in enumerate(imgs):
            imgs[i] = self.transform(img)
        actions = self._action_map(action)
        imgs = torch.stack(imgs)
        return imgs, torch.tensor(actions), 
    
    def _action_map(self, actions):
        updated_actions = []
        for action in actions:
            if action == "null":
                updated_actions.append(2)
            elif action == "false":
                updated_actions.append(0)
            elif action == "true":
                updated_actions.append(1)
        return updated_actions


# dts = DMSCtg_TaskDataset(stim_dir ='/home/xiaoxuan/projects/multfs_triple/Lucas_scripts/data/new_shapenet_val')
# img, action = dts[0]
# print(img.shape)

# for i in range(len(img)):
#     plt.figure()
        
#     plt.imshow(img[i])
#     plt.savefig("/home/xiaoxuan/projects/multfs_triple/sanity_check/fig_%d.png"%(i))
