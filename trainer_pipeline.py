from multiprocessing import pool
import time
import csv
import copy
import numpy as np

import json
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

from datetime import datetime


# create a folder named by experiment time
# folder include:
# configuration file
# logging folder
# plotted images
# saved checkpoint

# if not os.path.isdir("/home/xuan/projects/def-bashivan/xuan/multfs/triple_task/logdir/imagenette/"):
#     os.mkdir("/home/xuan/projects/def-bashivan/xuan/multfs/triple_task/logdir/imagenette/")
# logger = SummaryWriter("/home/xuan/projects/def-bashivan/xuan/multfs/triple_task/logdir/imagenette/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
def log_gradients_in_model(model, logger, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            # print(tag)
            # print(value.grad.cpu())
            logger.add_histogram(tag + "/grad", value.grad.cpu().type(torch.float64), step)
def bin2dec(input):
    input = list(input.type(torch.int64).cpu().numpy())
    temp = str("".join(map(str, input)))
    decimal_number = int(temp, 2)
    return decimal_number

    
from network.CNNCBNGRU import CNNCBNGRUNet
from network.CNNGRU import CNNGRUNet
from network.CNNLSTM import CNNLSTMNet
from network.CNNRNN import CNNRNNNet
from network.CNNGRU_avg_act import CNNGRUNetwithavgact
from network.CNNAttnLSTM import CNNAttnLSTMNet
from network.CNNconvLSTM import CNNConvLSTMNet
from network.convLSTM import ConvLSTMNet
from network.linearRNN import CNNlinearGRUNet
from network.CNNAttenPreLSTM import CNNAttenPreLSTMNet
from network.gpt2_embedding import gpt2_embedding
from tasks.nback_naturalistic import Nback_Dataset_Naturalistic
from tasks.interDMS_naturalistic import interDMS_Dataset_Naturalistic
from tasks.ctxDM_naturalistic import ctxDM_Dataset_Naturalistic
from tasks.multitask_naturalistic import Multitask_Dataset_Naturalistic


json_file = os.environ["JSONPATH"]


# why does te code only recognize --root but not other variables
parser = argparse.ArgumentParser()

# parser.add_argument('--json_file', '-jf', type=str, required=False, default = None )
# default = "/home/xuan/projects/def-bashivan/xuan/multfs/triple_task/config.json"
parser.add_argument('--root', '-r', type=str, required=True)


args = parser.parse_args()
root = args.root

with open(json_file, "r") as openfile:
    config_dict = json.load(openfile)

try:
    is_task_index_selfgen = config_dict["is_task_index_selfgen"]
    # n_tasks = config_dict["n_tasks"]
    n_tasks = 66
    
    task_index_shape = config_dict["task_index_shape"]


except: 
    is_task_index_selfgen = False
    n_tasks = 66 ######### that might be something to pay attention to with further modification to the code!!!!
try:
    is_optimal_task_index = config_dict["is_optimal_task_index"]
except:
    is_optimal_task_index = False
try: 
    is_cosannealing = config_dict["is_cosannealing"]
except:
    is_cosannealing = False
try:
    is_instruction = config_dict["is_instruction"]
except: is_instruction = False
print("is instruction?", is_instruction)


isload_model = config_dict["isload_model"]
loadmodel_path = config_dict["loadmodel_path"]
is_testing = config_dict["is_testing"]
taskname = config_dict["taskname"]
seq_len = config_dict["seq_len"]
is_fixed_feature = config_dict["is_fixed_feature"]
task_index_feature = config_dict["task_index_feature"]
dataset_type = config_dict["dataset_type"]
is_store = config_dict["is_store"]
is_noaction_loss = config_dict["is_noaction_loss"]
noaction_loss_weight = config_dict["noaction_loss_weight"]
if is_store:
    task_store_dir = os.path.join(root,"pd_1back_seq2_imagenette_small_ctg_whloc")
else:
    task_store_dir = root

try:
    epochs = config_dict["epochs"]
except: epochs = 1000

print(dataset_type)
# parameters for nback task
n_back_n_index = config_dict["nback"]
max_n = config_dict["max_n"]
# parameters for interDMS
task_index_feature_1 = config_dict["task_index_feature_1"]
task_index_feature_2 = config_dict["task_index_feature_2"]
is_fixed_pattern = config_dict["is_fixed_pattern"]
pattern = config_dict["pattern"]

# parameters for ctxDM
feature_pattern = config_dict["feature_pattern"]

# parameters for the network
try: is_RNN = config_dict["is_RNN"]
except: is_RNN = False
is_gru = config_dict["is_GRU"]
is_lstm = config_dict["is_LSTM"]
is_CBN = config_dict["is_CBN"]
is_resnet_pretrained = config_dict["is_resnet_pretrained"]
hidden_size = config_dict["hidden_size"]
is_avg_act = config_dict["is_avg_act"]
is_convlstm = config_dict["is_convlstm"]
if is_convlstm:
    is_gru = False
    is_CBN = False
is_pure_convlstm = config_dict["is_pure_convlstm"]
if is_pure_convlstm:
    is_gru = False
    is_CBN = False
is_attnprelstm = config_dict["is_attnprelstm"]
is_attnlstm = config_dict["is_attnlstm"]

# hyper params:
lr = config_dict["lr"]
gamma = config_dict["gamma"]

plt_model = config_dict["plt_model"]
is_noise = config_dict["is_noise"]
batch_size = config_dict["batch_size"]
train_dataset_size = config_dict["train_dataset_size"]

val_dataset_size = config_dict["val_dataset_size"]
label_size = config_dict["label_size"]
if is_task_index_selfgen:
    label_size = task_index_shape

currentDateAndTime = datetime.now()

saveroot = "/home/xuan/projects/def-bashivan/xuan/multfs/triple_task/experiment_logs/"

savefolder_0 = os.path.join(saveroot, "%d%d%d" % (currentDateAndTime.year, currentDateAndTime.month, currentDateAndTime.day,))
if not os.path.isdir(savefolder_0):
    os.mkdir(savefolder_0)

def init(rep):
    np.random.seed()
    random_int = np.random.randint(100, 999)

    save_folder = os.path.join(savefolder_0, "%d%d%d_rep%d_%d"%(currentDateAndTime.hour,currentDateAndTime.minute,currentDateAndTime.second, rep, random_int))
    

    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    shutil.copyfile(json_file, os.path.join(save_folder, "config.json")) # todo: need to specify configuration filepath, maybe from command line


    if not os.path.isdir(os.path.join(save_folder, "logdir")):
        os.mkdir(os.path.join(save_folder, "logdir"))
    logger = SummaryWriter(os.path.join(save_folder, "logdir"))
    return save_folder, logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if taskname == "nback":
    train_dataset = Nback_Dataset_Naturalistic(n = n_back_n_index, seq_len = seq_len, dataset_size = train_dataset_size, max_n = max_n, is_fixed_feature = is_fixed_feature, task_index_feature = task_index_feature, mode= "train", root = root, is_store = is_store, dataset_type = dataset_type, target_dir = task_store_dir, is_instruction = is_instruction, is_optimal_task_index=is_optimal_task_index)
    val_angle_dataset = Nback_Dataset_Naturalistic(n = n_back_n_index, seq_len = seq_len, dataset_size = val_dataset_size, max_n = max_n, is_fixed_feature = is_fixed_feature, task_index_feature = task_index_feature, mode= "val", root = root, is_store = is_store, dataset_type = dataset_type, target_dir = task_store_dir, is_instruction = is_instruction,is_optimal_task_index=is_optimal_task_index)
    val_new_obj_dataset = Nback_Dataset_Naturalistic(n = n_back_n_index, seq_len = seq_len, dataset_size = val_dataset_size, max_n = max_n, is_fixed_feature = is_fixed_feature, task_index_feature = task_index_feature, mode= "val_new_obj", root = root, is_store = is_store, dataset_type = dataset_type, target_dir = task_store_dir, is_instruction = is_instruction, is_optimal_task_index=is_optimal_task_index)
elif taskname == "ctxDM":
    train_dataset = ctxDM_Dataset_Naturalistic(dataset_size = train_dataset_size, is_fixed_feature = is_fixed_feature, feature_pattern = feature_pattern, mode= "train", root = root, dataset_type=dataset_type, is_instruction = is_instruction)
    val_angle_dataset = ctxDM_Dataset_Naturalistic(dataset_size = train_dataset_size, is_fixed_feature = is_fixed_feature, feature_pattern = feature_pattern, mode= "val", root = root,  dataset_type=dataset_type, is_instruction = is_instruction)
    val_new_obj_dataset = ctxDM_Dataset_Naturalistic(dataset_size = train_dataset_size, is_fixed_feature = is_fixed_feature, feature_pattern = feature_pattern, mode= "val_new_obj", root = root, dataset_type=dataset_type, is_instruction = is_instruction)   
elif taskname == "interDMS":
    train_dataset = interDMS_Dataset_Naturalistic(dataset_size = train_dataset_size, is_fixed_feature = is_fixed_feature, task_index_feature_1 = task_index_feature_1, task_index_feature_2 = task_index_feature_2, is_fixed_pattern = is_fixed_pattern, pattern = pattern, mode= "train", root = root, dataset_type=dataset_type,is_instruction = is_instruction )
    val_angle_dataset = interDMS_Dataset_Naturalistic(dataset_size = val_dataset_size, is_fixed_feature = is_fixed_feature, task_index_feature_1 = task_index_feature_1, task_index_feature_2 = task_index_feature_2, is_fixed_pattern = is_fixed_pattern, pattern = pattern, mode= "val", root = root, dataset_type=dataset_type, is_instruction = is_instruction)
    val_new_obj_dataset = interDMS_Dataset_Naturalistic(dataset_size = val_dataset_size, is_fixed_feature = is_fixed_feature, task_index_feature_1 = task_index_feature_1, task_index_feature_2 = task_index_feature_2, is_fixed_pattern = is_fixed_pattern, pattern = pattern, mode= "val_new_obj", root = root, dataset_type=dataset_type, is_instruction = is_instruction)
elif taskname == "multitask":
    train_dataset1 = Nback_Dataset_Naturalistic(n = n_back_n_index, seq_len = seq_len, dataset_size = train_dataset_size, max_n = max_n, is_fixed_feature = is_fixed_feature, task_index_feature = task_index_feature, mode= "train", root = root, is_store = is_store, dataset_type = dataset_type, target_dir = task_store_dir, is_instruction = is_instruction,is_optimal_task_index=is_optimal_task_index)
    train_dataset2 = ctxDM_Dataset_Naturalistic(dataset_size = train_dataset_size, is_fixed_feature = is_fixed_feature, feature_pattern = feature_pattern, mode= "train", root = root, dataset_type=dataset_type, is_instruction = is_instruction)
    train_dataset3 = interDMS_Dataset_Naturalistic(dataset_size = train_dataset_size, is_fixed_feature = is_fixed_feature, task_index_feature_1 = task_index_feature_1, task_index_feature_2 = task_index_feature_2, is_fixed_pattern = is_fixed_pattern, pattern = pattern, mode= "train", root = root, dataset_type=dataset_type, is_instruction = is_instruction)
    train_dataset = Multitask_Dataset_Naturalistic([train_dataset1, train_dataset2, train_dataset3],dataset_size = train_dataset_size, is_instruction = is_instruction)

    val_angle_dataset1 = Nback_Dataset_Naturalistic(n = n_back_n_index, seq_len = seq_len, dataset_size = val_dataset_size, max_n = max_n, is_fixed_feature = is_fixed_feature, task_index_feature = task_index_feature, mode= "val", root = root, is_store = is_store, dataset_type = dataset_type, target_dir = task_store_dir, is_instruction = is_instruction,is_optimal_task_index=is_optimal_task_index)
    val_angle_dataset2 = ctxDM_Dataset_Naturalistic(dataset_size = val_dataset_size, is_fixed_feature = is_fixed_feature, feature_pattern = feature_pattern, mode= "val", root = root,  dataset_type=dataset_type, is_instruction = is_instruction)
    val_angle_dataset3 = interDMS_Dataset_Naturalistic(dataset_size = val_dataset_size, is_fixed_feature = is_fixed_feature, task_index_feature_1 = task_index_feature_1, task_index_feature_2 = task_index_feature_2, is_fixed_pattern = is_fixed_pattern, pattern = pattern, mode= "val", root = root, dataset_type=dataset_type, is_instruction = is_instruction)
    val_angle_dataset = Multitask_Dataset_Naturalistic([val_angle_dataset1, val_angle_dataset2, val_angle_dataset3], dataset_size = val_dataset_size, is_instruction = is_instruction)

    val_new_obj_dataset1 = Nback_Dataset_Naturalistic(n = n_back_n_index, seq_len = seq_len, dataset_size = val_dataset_size, max_n = max_n, is_fixed_feature = is_fixed_feature, task_index_feature = task_index_feature, mode= "val_new_obj", root = root, is_store = is_store, dataset_type = dataset_type, target_dir = task_store_dir, is_instruction = is_instruction,is_optimal_task_index=is_optimal_task_index)
    val_new_obj_dataset2 = ctxDM_Dataset_Naturalistic(dataset_size = val_dataset_size, is_fixed_feature = is_fixed_feature, feature_pattern = feature_pattern, mode= "val_new_obj", root = root, dataset_type=dataset_type, is_instruction = is_instruction)   
    val_new_obj_dataset3 = interDMS_Dataset_Naturalistic(dataset_size = val_dataset_size, is_fixed_feature = is_fixed_feature, task_index_feature_1 = task_index_feature_1, task_index_feature_2 = task_index_feature_2, is_fixed_pattern = is_fixed_pattern, pattern = pattern, mode= "val_new_obj", root = root, dataset_type=dataset_type, is_instruction = is_instruction)
    val_new_obj_dataset = Multitask_Dataset_Naturalistic([val_new_obj_dataset1, val_new_obj_dataset2, val_new_obj_dataset3], dataset_size = val_dataset_size, is_instruction = is_instruction)



loaders = {"train": DataLoader(train_dataset, batch_size= batch_size, shuffle = True),
           "val_angle": DataLoader(val_angle_dataset, batch_size= batch_size, shuffle = True),
           "val_new_obj": DataLoader(val_new_obj_dataset, batch_size= batch_size, shuffle = True) # todo: change it back
           }





def name_generator(taskname, is_noise, is_CBN, is_resnet_pretrained, is_gru, hidden_size, seq_len, is_fixed_feature, 
                   is_convlstm = False, is_pure_convlstm = False, is_avg_act = False,
                    feature_index_1 = None, feature_index_2 = None, is_fixed_pattern = None, pattern = None,
                    feature_pattern = None,
                    feature_index=None, n_back_n_index = None,
                    is_testing = False, dataset_type = "imagenette"):
    if is_testing:
        name = "imagenette_testing_1back_category" # todo: change it back
        # name = "testing_1back_location"

    else:
        patterns = ["AABB", "ABAB","ABBA"]
        if taskname == "nback":
            name = "nback_%dback_seqlen%d_maxn%d_" % (n_back_n_index, seq_len, max_n)
            if is_fixed_feature:
                name += "feature%d_" % (feature_index)
            else: name += "multifeature_"
        elif taskname == "interDMS":
            if is_fixed_pattern:
                name = "interDMS_pattern%s_" % patterns[pattern]
            else: name = "interDMS_multipattern_"
            if is_fixed_feature:
                name += "feature%d%d_" % (feature_index_1, feature_index_2)
            else: name += "multifeature_"
        elif taskname == "ctxDM":
            if is_fixed_feature:
                name = "ctxDM_%s_"% feature_pattern
            else: name = "ctxDM_multipattern_"
        elif taskname == "multitask": # todo: to be further modified
            name = "multitask_"

        if is_noise:
            name += "noise_"
        if is_convlstm:
            name += "convlstm_"
        if is_pure_convlstm:
            name += "pure_convlstm_"
        if is_CBN:
            name += "CBN_"
        if is_resnet_pretrained:
            name += "resenetptr_"
        if is_gru:
            name += "gru_"
        if is_attnprelstm:
            name += "attnprelstm_"
        if is_attnlstm:
            name += "attnlstm"
        name += "hidden%d" % hidden_size
    if is_avg_act:
        name = "testing_imagenette_4ctg_loc_avg_act"
    name += "_%s" % dataset_type
    name += "_withactionloss" # todo: to be deleted
    # name += "_cls"
    # name = "testing_imagenette_4ctg_loc_convlstm"
    # name = "testing_shapenet_4loc_task_convlstm"
    return name

def training(rep, logger):
    if is_instruction: # decide the size of the instruction embeddings
        max_size = 0
        for i, input_collc in enumerate(loaders["train"]): # todo: to delete ctg_label
            print(i)
            input_img, action, n_index, instruction_old = input_collc
            
            # generate collections gpt2 embeddings
            diffs = []
            instruction = gpt2_embedding(list(instruction_old))
            
            max_size = instruction.shape[1]
            gpt2_instruction_embeddings = torch.zeros(n_tasks, max_size, 768)
            input_indices = [bin2dec(curr_index) for curr_index in n_index]
            for ti, input_index in enumerate(input_indices):
                gpt2_instruction_embeddings[input_index] = instruction[ti]

            if i == 2:
                print("max_size:", max_size)
                break
        
    name = name_generator(taskname=taskname, is_noise=is_noise, is_CBN=is_CBN, is_resnet_pretrained=is_resnet_pretrained, is_gru=is_gru, hidden_size=hidden_size, seq_len=seq_len, is_fixed_feature=is_fixed_feature, 
                            feature_pattern = feature_pattern, is_convlstm=is_convlstm, is_pure_convlstm=is_pure_convlstm, is_avg_act = is_avg_act,
                            feature_index_1 = task_index_feature_1, feature_index_2 = task_index_feature_2, is_fixed_pattern = is_fixed_pattern, pattern = pattern,
                            feature_index=task_index_feature, n_back_n_index = n_back_n_index,
                            is_testing = is_testing, dataset_type = dataset_type)

    if is_attnlstm:
        network = CNNAttnLSTMNet(hidden_size, output_size = 3,label_size = label_size).to(device)
    elif is_lstm:
        network = CNNLSTMNet(hidden_size=hidden_size, output_size = 3,label_size = label_size).to(device)
    elif is_attnprelstm:
        network = CNNAttenPreLSTMNet(hidden_size, output_size = 3,label_size = label_size).to(device)
    elif is_RNN:
        network = CNNRNNNet(hidden_size=hidden_size, output_size = 3,label_size = label_size).to(device)
        
    elif is_convlstm:
        network = CNNConvLSTMNet(hidden_size=hidden_size, output_size = 3,label_size = label_size).to(device)
    elif is_pure_convlstm:
        network = ConvLSTMNet(hidden_size=224, output_size = 3,label_size = label_size).to(device)
    elif is_avg_act:
        network = CNNGRUNetwithavgact(hidden_size=hidden_size, output_size = 3,label_size = label_size).to(device)
    else:
        if is_testing:
            network = CNNlinearGRUNet(hidden_size = hidden_size).to(device)
        else:
            if is_CBN:
                network = CNNCBNGRUNet(hidden_size=hidden_size, output_size=3,label_size = label_size,  batch_size = batch_size, size = 7, is_CBN = is_CBN).to(device)
            elif is_instruction:
                network = CNNGRUNet(hidden_size=hidden_size, output_size = 3,label_size = label_size, is_instruction = is_instruction, input_instruction_size = max_size).to(device)
            else: network = CNNGRUNet(hidden_size=hidden_size, output_size = 3,label_size = label_size, ).to(device)
    
    for layer_name, param in network.named_parameters():
        if param.requires_grad:
            # print(name, param.data)
            print(layer_name)
    
    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print("number of trainable parameters:", num_params)

    optimizer = optim.AdamW(network.parameters(), lr = lr )
    
    if is_task_index_selfgen:
        task_index_selfgen = nn.Embedding(n_tasks, task_index_shape)
        # new_group = {'params': task_index_selfgen, 'lr': lr}
        # optimizer.add_param_group(new_group)

       
    if is_cosannealing:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [100,200,300,400,500,], gamma = gamma)

    if isload_model:
        print("------------------------------------------is this loaded model?------------------------------------------------------------ ")
        checkpoint = torch.load(loadmodel_path)
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(checkpoint.keys())
        print(checkpoint["iters"])
        print(checkpoint["training_loss"])
        print(len(checkpoint["iters"]))
        print(len(checkpoint["training_loss"]))        
        


    # optimizer = optim.Adam(network.parameters(),lr=1e-1, weight_decay = 1e-3)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    criterion = nn.CrossEntropyLoss()
    name = name + "_rep%d"%rep
    print("model name:", name)
    
    ins_running_acc = 0
    ins_running_acc_action = 0
    ins_running_acc_noaction = 0
    if not isload_model:
        training_loss = []
        training_acc = []
        training_acc_action = []
        training_acc_noaction = []
        val_angle_acc = []
        val_angle_acc_action = []
        val_angle_acc_noaction = []
        val_angle_loss = []
        val_new_obj_acc = []
        val_new_obj_acc_action = []
        val_new_obj_acc_noaction = []
        val_new_obj_loss = []
        iters = []
        iters_val = []
        iters_val_new_obj = []
        iter = 0
        iter_val = 0
        iter_val_new_obj = 0
    else:
        iters = checkpoint["iters"]
        iter = iters[-1]
        iters_val = checkpoint["iters"].copy()
        iter_val = iters_val[-1]
        iters_val_new_obj = checkpoint["iters"].copy()
        iter_val_new_obj = iters_val_new_obj[-1]
        training_loss = checkpoint["training_loss"]
        training_acc = checkpoint["train_acc"]
        training_acc_action = checkpoint["train_acc_action"]
        training_acc_noaction = checkpoint["train_acc_noaction"]
        val_angle_loss = checkpoint["val_angle_loss"]
        val_angle_acc = checkpoint[ "val_angle_acc"]
        val_angle_acc_action = checkpoint["val_angle_acc_action"]
        val_angle_acc_noaction = checkpoint["val_angle_acc_noaction"]
        val_new_obj_loss = checkpoint["val_new_obj_loss"]
        val_new_obj_acc = checkpoint["val_new_obj_acc"]
        val_new_obj_acc_action = checkpoint["val_new_obj_acc_action"]
        val_new_obj_acc_noaction = checkpoint["val_new_obj_acc_noaction"]
    
    
    
    if isload_model:
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    if dataset_type == "shapenet":
        modes = ["train", "val_angle", "val_new_obj"]
    elif dataset_type == "imagenette":
        modes = ["train", "val_angle"]
    
    
    
    
    for epoch in range(start_epoch, epochs):
        for mode in modes:
            # print(mode)
            cumu_loss = 0
            correct = 0
            total = 0
            correct_action = 0
            correct_noaction = 0
            total_action = 0
            total_noaction = 0
            
            prev_best_running_acc = 0
            loader = loaders[mode]
            max_size = 0 # for task instruction embedding
            # print("dataloader before access:", datetime.now())
            

            print("new epoch!!!!")
            for i, input_collc in enumerate(loader): # todo: to delete ctg_label
                print("batch ",i)
                
                if not is_instruction:
            
                    input_img, action, n_index = input_collc
                else: 
                    input_img, action, n_index, instruction = input_collc

                    if max_size == 0:
                        # generate collections gpt2 embeddings
                        instruction = gpt2_embedding(list(instruction))
                        max_size = instruction.shape[1]
                        gpt2_instruction_embeddings = torch.zeros(n_tasks, max_size, 768)
                        input_indices = [bin2dec(curr_index) for curr_index in n_index]
                        for ti, input_index in enumerate(input_indices):
                            gpt2_instruction_embeddings[input_index] = instruction[ti]
                        print("instruction shape:", instruction.shape)
                        # print(gpt2_instruction_embeddings[input_indices])
                    else: 
                        # print(input_indices)
                        input_indices = [bin2dec(curr_index) for curr_index in n_index]
                        instruction = gpt2_instruction_embeddings[input_indices]
                    instruction = instruction.reshape(instruction.shape[0], -1)
                    
                
                if is_task_index_selfgen: 
                    # get all index
                    input_indices = torch.tensor([bin2dec(curr_index) for curr_index in n_index])
                    updated_n_index = task_index_selfgen(input_indices)
                              
                if i == 0:
                    logger.add_scalar('learning rate', scheduler.get_lr()[0], iter)
                if mode == "train":
                    network.train()
                    optimizer.zero_grad()
                else:
                    network.eval()
                
                # output = network(input_img,n_index, action = action, ctg_label = ctg_label) # todo: add noise
                if is_CBN:
                    
                    output = network(input_img, n_index)
                else:
                    if is_task_index_selfgen:
                        
                        output, action = network(input_img, updated_n_index, actions = action, ) # todo: to delete patching    
                        
                    elif is_instruction:
                        
                        output, aciton = network(input_img, instruction, actions = action)
                        
                    else:
                        output, action = network(input_img, n_index, actions = action, ) # todo: to delete patching
                if not is_noaction_loss:
                    action_index = [1]
                    no_action_index = [i for i in np.arange(2) if i not in action_index ]

                if not is_noaction_loss:
                    noaction_loss = criterion(output.permute(1,0,2)[:,no_action_index].reshape(-1,3), action[:,no_action_index].type(torch.LongTensor).to(device).reshape(-1))
                
                    action_loss = criterion(output.permute(1,0,2)[:,action_index].reshape(-1,3), action[:,action_index].type(torch.LongTensor).to(device).reshape(-1))

                # train_loss = action_loss + noaction_loss
                    train_loss = (1-noaction_loss_weight)*action_loss + noaction_loss_weight * noaction_loss
                    # print("here it goes!")
                else:
                    
                    train_loss = criterion(output.permute(1,0,2).reshape(-1,3), action.type(torch.LongTensor).to(device).reshape(-1))
                    
    
                cumu_loss += train_loss.item()
                
                
                if mode == "train":
                    train_loss.backward()
                    if i == 0 & iter // 1000 == 0:
                        log_gradients_in_model(network, logger, epoch)
                    optimizer.step()
                    
                    
                print("p4")
                _, predicted = torch.max(output.data, 2)
                predicted = predicted.permute(1,0)
                
                if not is_avg_act:
                    curr_correct = 0
                    total_action = 0
                    total_noaction = 0
                    curr_correct_noaction = 0
                    for t_i in range(action.shape[0]):
                            for t_j in range(action.shape[1]):
                                if action[t_i,t_j] != 2:
                                    total_action += 1
                                    if predicted[t_i,t_j] == action[t_i,t_j]:
                                        curr_correct += 1
                                else: 
                                    total_noaction += 1
                                    if predicted[t_i,t_j] == 2:
                                        curr_correct_noaction += 1
                    correct_action = curr_correct
                    correct_noaction = curr_correct_noaction
                else:
                    noaction_predicted = predicted[:,no_action_index]
                    action_predicted = predicted[:,action_index]
                    
                    noaction_action = action[:,no_action_index]
                    action_action = action[:, action_index] 
                    
                    total_action += len(action_predicted.reshape(-1))
                    
                    correct_action += (action_predicted.reshape(-1) == action_action.reshape(-1).to(device)).sum().item()

                    total_noaction += len(noaction_predicted.reshape(-1))
                    correct_noaction += (noaction_predicted.reshape(-1) == noaction_action.reshape(-1).to(device)).sum().item()
                    if iter == 0:
                        total += len(predicted.reshape(-1))
                        correct += (predicted.reshape(-1) == action.reshape(-1).to(device)).sum().item()

                        ins_running_acc = correct / total
                        ins_running_acc_action = correct_action / total_action
                        ins_running_acc_noaction = correct_noaction / total_noaction
                        if mode == "train":
                            training_acc.append(ins_running_acc)
                            training_acc_action.append(ins_running_acc_action)
                            training_acc_noaction.append(ins_running_acc_noaction)
                            training_loss.append(train_loss.data.cpu().numpy())
                            logger.add_scalar('training acc', ins_running_acc, iter)
                            logger.add_scalar('training acc action', ins_running_acc_action, iter)
                            logger.add_scalar('training acc no action', ins_running_acc_noaction, iter)
                            logger.add_scalar('training loss', train_loss.data.cpu().numpy(), iter)
                            if is_avg_act:
                                logger.add_scalar('training noaction loss', noaction_loss.data.cpu().numpy(), iter)
                                logger.add_scalar('training action loss', action_loss.data.cpu().numpy(), iter)
                # print("start of next epoch")
                if mode == "train":
                    iter += 1
                elif mode == "val_angle":
                    iter_val += 1
                elif mode == "val_new_obj":
                    iter_val_new_obj += 1
    

            if mode == "train":
                 iters.append(iter)        
            elif mode == "val_angle":
                iters_val.append(iter_val)
            elif mode == "val_new_obj":
                iters_val_new_obj.append(iter_val_new_obj)

            total += len(predicted.reshape(-1))
            correct += (predicted.reshape(-1) == action.reshape(-1).to(device)).sum().item()

            ins_running_acc = correct / total
            ins_running_acc_action = correct_action / total_action
            ins_running_acc_noaction = correct_noaction / total_noaction

            if mode == "train":
                training_acc.append(ins_running_acc)
                training_acc_action.append(ins_running_acc_action)
                training_acc_noaction.append(ins_running_acc_noaction)
                training_loss.append(train_loss.data.cpu().numpy())
                logger.add_scalar('training acc', ins_running_acc, iter)
                logger.add_scalar('training acc action', ins_running_acc_action, iter)
                logger.add_scalar('training acc no action', ins_running_acc_noaction, iter)
                logger.add_scalar('training loss', train_loss.data.cpu().numpy(), iter)
                if is_avg_act:
                    logger.add_scalar('training noaction loss', noaction_loss.data.cpu().numpy(), iter)
                    logger.add_scalar('training action loss', action_loss.data.cpu().numpy(), iter)

            elif mode == "val_angle":
                val_angle_acc.append(ins_running_acc)
                val_angle_acc_action.append(ins_running_acc_action)
                val_angle_acc_noaction.append(ins_running_acc_noaction)
                val_angle_loss.append(train_loss.data.cpu().numpy())
                logger.add_scalar('val acc', ins_running_acc, iter)
                logger.add_scalar('val acc action', ins_running_acc_action, iter)
                logger.add_scalar('val acc no action', ins_running_acc_noaction, iter)
                logger.add_scalar('val loss', train_loss.data.cpu().numpy(), iter)
                if is_avg_act:
                    logger.add_scalar('val noaction loss', noaction_loss.data.cpu().numpy(), iter)
                    logger.add_scalar('val action loss', action_loss.data.cpu().numpy(), iter)
            elif mode == "val_new_obj":
                val_new_obj_acc.append(ins_running_acc)
                val_new_obj_acc_action.append(ins_running_acc_action)
                val_new_obj_acc_noaction.append(ins_running_acc_noaction)
                val_new_obj_loss.append(train_loss.data.cpu().numpy())
                logger.add_scalar('val new obj acc', ins_running_acc, iter)
                logger.add_scalar('val new obj acc action', ins_running_acc_action, iter)
                logger.add_scalar('val new obj acc no action', ins_running_acc_noaction, iter)
                logger.add_scalar('val new obj loss', train_loss.data.cpu().numpy(), iter)
                if is_avg_act:
                    logger.add_scalar('val new obj noaction loss', noaction_loss.data.cpu().numpy(), iter)
                    logger.add_scalar('val new obj action loss', action_loss.data.cpu().numpy(), iter)
            
            if mode == "train":
                scheduler.step()
                print(name, "current epoch is %d, %s loss is %.2f, %s accuracy is %f" % (epoch, mode, train_loss, mode, ins_running_acc))
                print("action accuracy is %f, noaction accuracy is %f" % (ins_running_acc_action, ins_running_acc_noaction))
            if (dataset_type == "shapenet" and mode == "val_new_obj") or (dataset_type == "imagenette" and mode == "val_angle"):
                # save the model
                if prev_best_running_acc < training_acc[-1]:
                    print("save the model")
                    
                    savename = os.path.join(save_folder, "checkpoint.pth")
                    # print("optimizer state keys")
                    # state_dict_keys = optimizer.state_dict().keys()
                    # print(state_dict_keys)
                    # print("--------------------------")
                    save_dict = {'epoch':epoch,
                                 'iters': iters,
                                "model_state_dict": network.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "training_loss": training_loss,
                                "train_acc": training_acc,
                                "train_acc_action": training_acc_action,
                                "train_acc_noaction": training_acc_noaction,
                                "val_angle_loss": val_angle_loss,
                                "val_angle_acc": val_angle_acc,
                                "val_angle_acc_action": val_angle_acc_action,
                                "val_angle_acc_noaction": val_angle_acc_noaction,
                                "val_new_obj_loss": val_new_obj_loss,
                                "val_new_obj_acc": val_new_obj_acc,
                                "val_new_obj_acc_action": val_new_obj_acc_action,
                                "val_new_obj_acc_noaction": val_new_obj_acc_noaction,
                                }

                    if is_task_index_selfgen:
                        save_dict["task_index_selfgen"] = task_index_selfgen
                    torch.save(save_dict, savename)
                   
                    # save the figure
                    print("save the figure")
                   
                    plt.figure()
                    plt.plot(iters[-len(training_loss):], training_loss, label ="train")
                    plt.plot(iters[-len(val_angle_loss):], val_angle_loss, label ="val")
                    if dataset_type == "shapenet":
                        plt.plot(iters[-len(val_new_obj_loss):], val_new_obj_loss, label ="val_new_obj")
                    plt.legend()
                    plt.xlabel("iters")
                    plt.ylabel("loss")
                    
                    plt.title("%s loss" % name)
                    plt.savefig(os.path.join(save_folder, "loss.png"))

                    plt.figure()
                    plt.plot(iters[-len(training_acc):],training_acc, label = "train")
                    plt.plot(iters[-len(val_angle_acc):],val_angle_acc, label = "val")
                    if dataset_type == "shapenet":
                        plt.plot(iters[-len(val_new_obj_acc):], val_new_obj_acc, label = "val_new_obj")
                    plt.legend()
                    plt.xlabel("iters")
                    plt.ylabel("accuracy_level")
    
                    plt.title("%s accuracy" % name)
                    plt.savefig(os.path.join(save_folder, "acc.png"))
                    
                    prev_best_running_acc = training_acc[-1]

                    plt.figure()
                    plt.plot(iters[-len(training_acc_action):], training_acc_action, label = "train_act")
                    plt.plot(iters[-len(val_angle_acc_action):], val_angle_acc_action, label = "val_act")
                    if dataset_type == "shapenet":
                        plt.plot(iters[-len(val_new_obj_acc_action):],val_new_obj_acc_action, label = "val_new_obj_act")
                    plt.plot(iters[-len(training_acc_noaction):],training_acc_noaction, "--", label = "train_noact")
                    plt.plot(iters[-len(val_angle_acc_noaction):],val_angle_acc_noaction,"--", label = "val_noact")
                    if dataset_type == "shapenet":
                        plt.plot(iters[-len(val_new_obj_acc_noaction):],val_new_obj_acc_noaction,"--", label = "val_new_obj_noact")
                    plt.legend()
                    plt.xlabel("iters")
                    plt.ylabel("accuracy_level")
    
                    plt.title("%s accuracy" % name)
                    plt.savefig(os.path.join(save_folder, "acc_wthaction.png"))
                



if __name__ == '__main__':
    # multithread training
    print("before training:", datetime.now())

    # reps = np.arange(2)
    # loggers = []
    # for rep in np.arange(reps):
    #     save_folder, logger = init(rep)
    #     loggers.append(logger)

    
    # with multiprocessing.Pool() as pool:
    #     # Use the pool to apply the shuffle_array function to the array
        
    #     pool.apply_async(training, args = (reps[i], loggers[i]) for i in np.arange(10))

        
    rep_list = np.arange(1)

    
    for rep in rep_list:
        save_folder, logger = init(rep)
        print("current replication is:", rep)
        training(rep, logger)




