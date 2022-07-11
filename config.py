from utils.indices2coordinates import indices2coordinates
from utils.compute_window_nums import compute_window_nums
import numpy as np

CUDA_VISIBLE_DEVICES = '0'  # The current version only supports one GPU training


set = 'wildbees'  # Different dataset with different
model_name = 'resnet50'

batch_size = 64
vis_num = batch_size * 5 # The number of visualized images in tensorboard
eval_trainset = True  # Whether or not evaluate trainset
save_interval = 1
max_checkpoint_num = 5 #200
end_epoch = 5 #200
init_lr = 0.001
lr_milestones = [2, 4] #[60, 100]
lr_decay_rate = 0.1
weight_decay = 1e-4
stride = 32
channels = 2048 # num. of neurons in the Dense layer (fixed for ResNet50)
input_size = 224#448

# The pth path of pretrained model
pretrain_path = './models/pretrained/resnet50-19c8e357.pth'

if set == 'CUB':
    model_path = './checkpoint/cub'  # pth save path
    root = './datasets/CUB_200_2011'  # dataset path
    num_classes = 200
    # windows info for CUB
    N_list = [2, 3, 2]
    window_side = [128, 192, 256]
    
    ratios = [[4, 4], [3, 5], [5, 3],
              [6, 6], [5, 7], [7, 5],
              [8, 8], [6, 10], [10, 6], [7, 9], [9, 7], [7, 10], [10, 7]]
    
else:   
    ratios = [[6, 6], [5, 7], [7, 5],
              [8, 8], [6, 10], [10, 6], [7, 9], [9, 7],
              [10, 10], [9, 11], [11, 9], [8, 12], [12, 8]]
    
    if set == 'CAR':
        N_list = [3, 2, 1]
        window_side = [192, 256, 320]
        model_path = './checkpoint/car'      # pth save path
        root = './datasets/Stanford_Cars'  # dataset path
        num_classes = 196
    elif set == 'Aircraft':
        N_list = [3, 2, 1]
        window_side = [192, 256, 320]
        model_path = './checkpoint/aircraft'      # pth save path
        root = './datasets/FGVC-aircraft'  # dataset path
        num_classes = 100
    elif set == 'iNat':
        N_list = [1, 1, 1]        
        window_side = [128, 192, 256]
        model_path = './checkpoint/inat'      # pth save path
        root = './datasets/iNat'  # dataset path
        num_classes = 37
    elif set == 'iNat_3spec':
        N_list = [3, 2, 1]        
        window_side = [64, 128, 192]
        model_path = './checkpoint/inat_3spec'      # pth save path
        root = './datasets/iNat_3spec'  # dataset path
        num_classes = 3
    elif set == 'wildbees':
        N_list = [3, 2, 1]
        window_side = [64, 128, 192]
        model_path = './checkpoint/wildbees'      # pth save path
        root = '../..'  # dataset path
        num_classes = 22


proposalN = sum(N_list)  # proposal window num
iou_threshs = [0.25] * len(N_list)


'''indice2coordinates'''
window_nums = compute_window_nums(ratios, stride, input_size)
indices_ndarrays = [np.arange(0,window_num).reshape(-1,1) for window_num in window_nums]
coordinates = [indices2coordinates(indices_ndarray, stride, input_size, ratios[i]) for i, indices_ndarray in enumerate(indices_ndarrays)] # 每个window在image上的坐标
coordinates_cat = np.concatenate(coordinates, 0)
window_milestones = [sum(window_nums[:i+1]) for i in range(len(window_nums))]
if set == 'CUB' or set == 'iNat_3spec':
    window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:6]), sum(window_nums[6:])]
else:
    window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:8]), sum(window_nums[8:])]
