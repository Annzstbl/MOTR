# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
PWD=$(cd `dirname $0` && pwd)
cd $PWD/../

PRETRAIN=/data/users/litianhao/hsmot_code/workdir/motr/motr_r50_train_hsmot_8ch_4gpu/checkpoint0019.pth

EXP_DIR=/data/users/litianhao/hsmot_code/workdir/motr/motr_r50_train_hsmot_8ch_4gpu
mkdir -p ${EXP_DIR}
touch ${EXP_DIR}/predict.log
cp $0 ${EXP_DIR}/

CUDA_VISIBLE_DEVICES=0 python3 eval_hsmot_8ch.py \
    --pretrained ${PRETRAIN} \
    --output_dir ${EXP_DIR} \
    --mot_path /data/users/litianhao/data/HSMOT \
    --resume ${PRETRAIN} \
    --run_time_tracker_th1 0.6 \
    --meta_arch motr \
    --use_checkpoint \
    --dataset_file e2e_hsmot_8ch \
    --epoch 20 \
    --with_box_refine \
    --lr_drop 10 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
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
    --num_workers 0 \
    | tee ${EXP_DIR}/predict.log
