# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
PWD=$(cd `dirname $0` && pwd)
cd $PWD/../

PRETRAIN=/data3/litianhao/hsmot/motr/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint_8ch_interpolate.pth
EXP_DIR=/data3/litianhao/hsmot/motr/99/3ch_rotateAttn_1gpu
mkdir -p ${EXP_DIR}
touch ${EXP_DIR}/output.log
cp $0 ${EXP_DIR}/

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 \
CUDA_VISIBLE_DEVICES=3 python3 -m torch.distributed.launch --nproc_per_node=1 \
    --master_port 20012 \
    --use_env main_8ch.py \
    --meta_arch motr \
    --use_checkpoint \
    --dataset_file e2e_hsmot_8ch \
    --epoch 20 \
    --with_box_refine \
    --lr_drop 10 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained ${PRETRAIN} \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 3 \
    --sampler_steps 5 9 15 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --mot_path /data/users/wangying01/lth/hsmot/data/HSMOT \
    --num_workers 2 \
    --input_channels 3 \
    --npy2rgb \
    | tee ${EXP_DIR}/output.log -a
