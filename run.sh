#!/bin/bash

repo_dir=/mnt/Data2/liyan/online_lang_splatting

#python3 eval/create_replica_labels.py --langslam_dir ${repo_dir}/results/replica/room0/omni_data_result/imap/2026-07-03-16-59-13/psnr/before_opt \
#                                       --seg_file_config /mnt/Data4/slam_datasets/vmap/room_0/imap/01/render_config.yaml

python3 eval/evaluate_onlinelangslam.py --dataset_name room0 \
  --root_dir ${repo_dir}/results/replica/room0/omni_data_result/imap/2026-07-03-16-59-13/psnr/before_opt \
  --ae_ckpt_dir ${repo_dir}/data/OnlineLanguageSplatting/Pretrained_models/omni_general/ae_149_he.ckpt \
  --online_ae_ckpt ${repo_dir}/results/replica/room0/omni_data_result/online_15_room0.pth