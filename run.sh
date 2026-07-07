#!/bin/bash

repo_dir=/mnt/Data2/liyan/online_lang_splatting

# --- Single-stage (disentanglement) SLAM run output to evaluate ---
run_dir=${repo_dir}/results/replica/room0/disent_result/slam_datasets/2026-07-06-19-09-54/psnr/before_opt

# Replica GT segmentation config (vmap dataset)
seg_config=/mnt/Data4/slam_datasets/vmap/room_0/imap/01/render_config.yaml

# Pretrained single-stage disentanglement autoencoder (code size 3)
ae_ckpt=${repo_dir}/data/OnlineLanguageSplatting/Pretrained_models/single_stage_AE_disent.pth

# --- Step 1: build GT label .json files from Replica segmentation masks ---
python3 eval/create_replica_labels.py \
  --langslam_dir ${run_dir} \
  --seg_file_config ${seg_config}

# --- Step 2: evaluate single-stage (disentanglement) results ---
# Single-stage disent AE: code size 3, 7-layer encoder
python3 eval/evaluate_langslam.py --dataset_name room0 \
  --root_dir ${run_dir} \
  --ae_ckpt_dir ${ae_ckpt} \
  --code_size 3 \
  --encoder_dims 384 192 96 48 24 12 3 \
  --decoder_dims 12 24 48 96 192 384 768

# --- (reference) 2-stage eval, not used for single-stage ---
# python3 eval/evaluate_onlinelangslam.py --dataset_name room0 \
#   --root_dir ${repo_dir}/results/replica/room0/omni_data_result/imap/<TIMESTAMP>/psnr/before_opt \
#   --ae_ckpt_dir ${repo_dir}/data/OnlineLanguageSplatting/Pretrained_models/omni_general/ae_149_he.ckpt \
#   --online_ae_ckpt ${repo_dir}/results/replica/room0/omni_data_result/online_15_room0.pth
