PWD=$(cd `dirname $0` && pwd)
cd $PWD/../

PRETRAIN=/data/users/litianhao/hsmot_code/workdir/motr/20241230_e2e_motr_r50_train_hsmot_rgb_01_l1_mmrotate_interval3/checkpoint.pth
EXP_DIR=/data/users/litianhao/hsmot_code/workdir/motr/20241230_e2e_motr_r50_train_hsmot_rgb_01_l1_mmrotate_interval3


CUDA_VISIBLE_DEVICES=4 python3 eval_hsmot_rgb.py \
    --meta_arch motr \
    --use_checkpoint \
    --dataset_file e2e_hsmot_rgb \
    --epoch 20 \
    --with_box_refine \
    --lr_drop 10 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained ${PRETRAIN} \
    --resume ${PRETRAIN} \
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
    --mot_path /data3/PublicDataset/Custom/HSMOT 
