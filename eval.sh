#!/usr/bin/env bash

gpus=0

data_name=PVPanel-CD
net_G=ChangeAdapter #This is the best version
# net_G=ChangeFormerV6 #This is the best version
split=val
vis_root=vis
project_name=1024dim
checkpoints_root=outputs
checkpoint_name=iou9246.pt
img_size=512
embed_dim=1024 #Make sure to change the embedding dim (best and default = 256)

CUDA_VISIBLE_DEVICES=0 python eval_cd.py --split ${split} --net_G ${net_G} --embed_dim ${embed_dim} --img_size ${img_size} --vis_root ${vis_root} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}



