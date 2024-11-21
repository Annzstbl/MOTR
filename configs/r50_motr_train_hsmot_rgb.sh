# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


# for MOT17

# PRETRAIN=/data/users/litianhao/hsmot_code/workdir/motr/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth
# EXP_DIR=/data/users/litianhao/hsmot_code/workdir/motr/e2e_motr_r50_train_hsmot_rgb
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 \
#     --use_env main.py \
#     --meta_arch motr \
#     --use_checkpoint \
#     --dataset_file e2e_hsmot_rgb \
#     --epoch 20 \
#     --with_box_refine \
#     --lr_drop 10 \
#     --lr 2e-4 \
#     --lr_backbone 2e-5 \
#     --pretrained ${PRETRAIN} \
#     --output_dir ${EXP_DIR} \
#     --batch_size 1 \
#     --sample_mode 'random_interval' \
#     --sample_interval 10 \
#     --sampler_steps 5 9 15 \
#     --sampler_lengths 2 3 4 5 \
#     --update_query_pos \
#     --merger_dropout 0 \
#     --dropout 0 \
#     --random_drop 0.1 \
#     --fp_ratio 0.3 \
#     --query_interaction_layer 'QIM' \
#     --extra_track_attn \
#     --data_txt_path_train ./datasets/data_path/joint.train \
#     --data_txt_path_val ./datasets/data_path/mot17.train \
#     --mot_path /data3/PublicDataset/Custom/HSMOT \
#     | tee ${EXP_DIR}/output.log
PRETRAIN=/data/users/litianhao/hsmot_code/workdir/motr/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
EXP_DIR=/data/users/litianhao/hsmot_code/workdir/motr/e2e_motr_r50_train_hsmot_rgb_debug
mkdir ${EXP_DIR}
touch ${EXP_DIR}/output.log
CUDA_VISIBLE_DEVICES=0,3 python3 -m torch.distributed.launch --nproc_per_node=2 \
    --master_port 20011 \
    --use_env main.py \
    --meta_arch motr \
    --use_checkpoint \
    --dataset_file e2e_hsmot_rgb \
    --epoch 20 \
    --with_box_refine \
    --lr_drop 10 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained ${PRETRAIN} \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 10 \
    --sampler_steps 5 9 15 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --data_txt_path_train ./datasets/data_path/joint.train \
    --data_txt_path_val ./datasets/data_path/mot17.train \
    --mot_path /data3/PublicDataset/Custom/HSMOT \
    --num_workers 4 \
    | tee ${EXP_DIR}/output.log
