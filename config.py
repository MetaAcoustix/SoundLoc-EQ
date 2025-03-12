"""
@File: config
@Time: 3/12/25 10:19 PM
@Author: liincuan
@Description: 
"""
seed = 42

model_type = 'dnn'

epochs_max = 30
gpus = [5]
# gpus = [0,1,2,3,4,5,6,7]

angle_range = 360
cell_reso = 72
cell_len = angle_range / cell_reso
sigma = 8
alpha = 0.2

space = 'circular'

strategy = "ddp"

learning_rate = 1e-3            # initial
min_lr = learning_rate / 10
scheduler_factor = 0.1
patience = 3                    # the epoch number of learning rate decay
patience_stop = 10              # traing terminated prematurely
