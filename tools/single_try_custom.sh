#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=0 python /home/whua/code/ie_e2e/tools/train.py '/home/whua/code/ie_e2e/configs/vie_custom/e2e_vie/default_config_e2e.py' \
#--work-dir='/data/whua/ie_e2e_log/test_run'
#CUDA_VISIBLE_DEVICES=0 python /home/whua/code/ie_e2e/tools/train.py '/home/whua/code/ie_e2e/configs/vie_custom/e2e_vie/asyncreader_local_config_e2e.py' \
#--work-dir='/data/whua/ie_e2e_log/test_run'
# 1032 server
## eval
#CUDA_VISIBLE_DEVICES=6 python /home/whua/project/ie_e2e/tools/test.py \
#'/home/whua/project/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_ar_1280_1e4_both_rec80_nonode_adam_cloud.py' \
#'/home/whua/logs/ie_weights/sroie_finetune_both_ocr80_kie80_nonode_adam_1e4_240.pth' --eval 'sroie' \
#--show-dir='/home/whua/logs/ie_e2e_log/eval_vis_sroie_nonode'
## SROIE
#CUDA_VISIBLE_DEVICES=0 python /home/whua/project/ie_e2e/tools/train.py \
#'/home/whua/project/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_ar_1280_4e4_both_fuse_local_1032.py' \
#--work-dir='/home/whua/logs/ie_e2e_log/single_run' --deterministic
## SROIE-Multi-GPU
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 --master_port=29500 \
#/home/whua/project/ie_e2e/tools/train.py \
#/home/whua/project/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_ar_1280_1e4_both_fuse_local_1032.py \
#--work-dir=/home/whua/logs/ie_e2e_log/sroie_1e4_both_feature_fuse_sort --launcher pytorch --gpus 6 \
#--deterministic
# EPHOIE
#CUDA_VISIBLE_DEVICES=0 python /home/whua/project/ie_e2e/tools/train.py \
#'/home/whua/project/ie_e2e/configs/vie_custom/e2e_ar_vie/ephoie/ephoie_ar_1280_1e4_both_rec80_gca_local_1032.py' \
#--work-dir='/home/whua/logs/ie_e2e_log/single_run'
#CUDA_VISIBLE_DEVICES=0 python /home/whua/project/ie_e2e/tools/train.py \
#'/home/whua/project/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/ephoie/ephoie_ar_1280_1e4_ocr40_adam_local_1032.py' \
#--work-dir='/home/whua/logs/ie_e2e_log/single_run' --deterministic
## ----------------- 1032 server -----------------
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
#/home/whua/project/ie_e2e/tools/train.py \
#/home/whua/project/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/ephoie/local/ephoie_sdef_nark_4l_noalign_local_400e_1280_1032.py \
#--work-dir=/home/whua/logs/ie_ocr_log/ephoie_edef_SEED_nark_4l_noalign_local_400e_1280_bs4 --launcher pytorch --gpus 4 \
#--load-from=/home/jfkuang/code/ie_e2e/pretrained_models/epoch_10.pth \
#--deterministic --seed 1364371869
# nfv5_3125_sdef_SEED_nark_3l_kvc_KR_SELFATTN_1STG_200e_720_pt_bs4
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
#/home/whua/project/ie_e2e/tools/train.py \
#/home/whua/project/ie_e2e/configs/vie_custom/e2e_trie/sroie/local/sroie_sc_sdef_norot_gt_200e_local_1032.py \
#--work-dir=/home/whua/logs/ie_e2e_log/trie_sroie_sdef_norot_gt_200e_bs4 --launcher pytorch --gpus 4 \
#--load-from=/home/whua/logs/ie_weights/sroie_FORMAL_3L_screen_ft_default_dp02_rr_3l_lr2e4_bs4_epoch_570.pth \
#--deterministic
#CUDA_VISIBLE_DEVICES=6 python /home/whua/project/ie_e2e/tools/train.py \
#/home/whua/project/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_sdef_rr_nark_3l_rec1_wrec_local_1032.py \
#--work-dir=/home/whua/logs/ie_e2e_log/single_test \
#--load-from=/home/whua/logs/ie_e2e_log/sroie_sdef_rr_nark_3l_rec1_kvc_KR_SELFATTN_1STG_200e_bs4/latest.pth \
#--deterministic --seed 1364371869
# ---------------- 1061 server -------------
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
/home/whua/code/ie_e2e/tools/train.py \
/home/whua/code/ie_e2e/configs/vie_custom/e2e_trie/v5/local/nfv5_3125_sdef_3l_200e_720_local_1061.py \
--work-dir=/home/whua/logs/ie_e2e_log_0902/trie_v5_3125_sdef_rr_nark_3l_rec1_200e_bs4 --launcher pytorch --gpus 4 \
--load-from=/home/whua/logs/ie_weights/ocr_pretrain_eng_full_default_dp02_rc_rr_cj_blsh_lr4e4_dpp02_epoch_6.pth \
--deterministic --seed 1364371869
# --load-from=/home/whua/logs/ie_weights/ocr_pretrain_eng_full_default_dp02_rc_rr_cj_blsh_lr4e4_dpp02_epoch_6.pth
# --load-from=/home/jfkuang/code/ie_e2e/pretrain_models/epoch_10.pth \
#CUDA_VISIBLE_DEVICES=0 python /home/whua/code/ie_e2e/tools/train.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_nark_nomsk_3l_200e_720_local_1061.py \
#--work-dir=/home/whua/logs/ie_ocr_log/single_run \
#--deterministic
# ---------------- 9999 server -------------
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 \
#/home/whua/code/ie_e2e/tools/train.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_sdef_rr_nark_3l_rec1_commic_drec_200e_local_9999.py \
#--work-dir=/data/whua/ie_e2e_log_0516/sroie_sdef_SEED_rr_nark_3l_rec1_commic_drec_200e_bs4 --launcher pytorch --gpus 2 \
#--load-from=/data/whua/ie_e2e_weights/sroie_FORMAL_3L_screen_ft_default_dp02_rr_3l_lr2e4_bs4_epoch_570.pth \
#--deterministic --seed 1364371869
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 \
#/home/whua/code/ie_e2e/tools/train.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_sdef_rr_nark_3l_rec1_KV_KRFeat_1STG_200e_cloud_tal.py \
#--work-dir=/data/whua/ie_e2e_log_0516/sroie_sdef_rr_nark_3l_rec1_KV_KRFeat_1STG_200e_bs2 --launcher pytorch --gpus 2 \
#--load-from=/data/whua/ie_e2e_weights/sroie_FORMAL_3L_screen_ft_default_dp02_rr_3l_lr2e4_bs4_epoch_570.pth \
#--deterministic --seed 1364371869
#CUDA_VISIBLE_DEVICES=1 python /home/whua/code/ie_e2e/tools/train.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/custom_dataset/custom_chn_v2_sdef_dp02_lr4e4_dpp02_cloud.py \
#--work-dir=/data/whua/ie_ocr_log/single_test_chn \
#--load-from=/data/whua/ie_e2e_weights/sroie_FORMAL_3L_screen_ft_default_dp02_rr_3l_lr2e4_bs4_epoch_570.pth \
#--deterministic
# ---------------- 1803 server -------------
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=27500 \
#/home/whua/code/ie_e2e/tools/train.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_sdef_rr_nark_3l_rec1_200e_local_1803.py \
#--work-dir=/home/whua/logs/ie_e2e_log_0902/sroie_noinplace_sdef_rr_nark_3l_rec1_200e_bs4 --launcher pytorch --gpus 4 \
#--resume-from=/home/whua/logs/ie_e2e_log_0902/sroie_noinplace_sdef_rr_nark_3l_rec1_200e_bs4/latest.pth \
#--deterministic --seed 1364371869
# --load-from=/home/whua/logs/ie_ocr_weights/sroie_FORMAL_3L_screen_ft_default_dp02_rr_3l_lr2e4_bs4_epoch_570.pth
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=28500 \
#/home/whua/code/ie_e2e/tools/train.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_nark_3l_seq_wkie_200e_720_local_1803.py \
#--work-dir=/home/whua/logs/ie_e2e_log/v5_3125_sdef_SEED_noinplace_nark_3l_seq_wkie_200e_720_local_200e_720_bs4 --launcher pytorch --gpus 4 \
#--load-from=/home/whua/logs/ie_ocr_weights/ocr_pretrain_eng_full_default_dp02_rc_rr_cj_blsh_lr4e4_dpp02_epoch_6.pth \
#--deterministic --seed 1364371869
#CUDA_VISIBLE_DEVICES=4 python /home/whua/code/ie_e2e/tools/train.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_nark_3l_seq_200e_720_local_1803.py \
#--work-dir=/home/whua/logs/ie_ocr_log/single_test \
#--load-from=/home/whua/logs/ie_e2e_log/v5_3125_sdef_SEED_nark_3l_cuskvc_KR_1STG_200e_720_bs4/latest.pth \
#--deterministic
# -------- 10081 ---------
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 \
#/home/whua/code/ie_e2e/tools/train.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_screen_serial_sdef_lncrf_rr_nark_rec1_glb_200e_local_10081.py \
#--work-dir=/mnt/whua/logs/ie_e2e_log/sroie_sroie_def_rr_nark_rec1_glb_200e_bs2 --launcher pytorch --gpus 2 \
#--load-from=/mnt/whua/model_weights/sroie_FORMAL_screen_ft_default_dp02_lr2e4_rr_bs4_epoch_300.pth \
#--deterministic
#CUDA_VISIBLE_DEVICES=0 python /home/whua/code/ie_e2e/tools/train.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_screen_serial_sdef_lncrf_rr_nark_enckie_l2ks_200e_local_10081.py \
#--work-dir=/mnt/whua/logs/ie_e2e_log/single_test \
#--load-from=/mnt/whua/model_weights/sroie_FORMAL_screen_ft_default_dp02_lr2e4_rr_bs4_epoch_300.pth \
#--deterministic
# --------- 1031 ---------
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
#/home/whua/code/ie_e2e/tools/train.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_sdef_rr_nark_3l_rec1_seq_wkie_200e_local_1031.py \
#--work-dir=/mnt/whua/logs/ie_e2e_log/sroie_sdef_SEED_rr_nark_3l_rec1_seq_wkie_200e_bs4 --launcher pytorch --gpus 4 \
#--resume-from=/mnt/whua/logs/ie_e2e_log/sroie_sdef_SEED_rr_nark_3l_rec1_seq_wkie_200e_bs4/latest.pth \
#--deterministic --seed 1364371869
# --load-from=/mnt/whua/logs/ie_e2e_weights/sroie_FORMAL_3L_screen_ft_default_dp02_rr_3l_lr2e4_bs4_epoch_570.pth
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
#/home/whua/code/ie_e2e/tools/train.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_screen_serial_sdef_lncrf_rr_nark_3l_rec1_seq_200e_local_1031.py \
#--work-dir=/mnt/whua/logs/ie_e2e_log/sroie_def_rr_nark_3l_rec1_seq_200e_bs4 --launcher pytorch --gpus 4 \
#--load-from=/mnt/whua/logs/ie_e2e_weights/sroie_FORMAL_3L_screen_ft_default_dp02_rr_3l_lr2e4_bs4_epoch_570.pth \
#--deterministic
#CUDA_VISIBLE_DEVICES=0 python /home/whua/code/ie_e2e/tools/train.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_screen_serial_sdef_lncrf_rr_nark_3l_rec1_seq_200e_local_1031.py \
#--work-dir=/mnt/whua/logs/ie_e2e_log/single_test \
#--load-from=/mnt/whua/logs/ie_e2e_weights/sroie_FORMAL_3L_screen_ft_default_dp02_rr_3l_lr2e4_bs4_epoch_570.pth \
#--deterministic
# --load-from=/mnt/whua/logs/ie_e2e_weights/sroie_FORMAL_screen_ft_default_dp02_lr2e4_rr_bs4_epoch_300.pth
# --------- 1062 ----------
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
#/home/whua/code/ie_e2e/tools/train.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_nark_3l_rec1_cuskvc_KR_1STG_200e_720_local_1062.py \
#--work-dir=/home/whua/logs/ie_e2e_log/nfv5_3125_sdef_SEED_nark_3l_rec1_cuskvc_KR_1STG_200e_720_bs4 --launcher pytorch --gpus 4 \
#--resume-from=/home/whua/logs/ie_e2e_log/nfv5_3125_sdef_SEED_nark_3l_rec1_cuskvc_KR_1STG_200e_720_bs4/latest.pth \
#--deterministic --seed 1364371869
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
#/home/whua/code/ie_e2e/tools/train.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_trie/v5_2200/local/nfv5_2200_3128_sdef_gt_200e_720_local_1062.py \
#--work-dir=/home/whua/logs/ie_e2e_log/trie_v5_2200_3128_GT_200e_720_bs4 --launcher pytorch --gpus 4 \
#--load-from=/home/whua/logs/ie_e2e_weights/ocr_pretrain_eng_full_default_dp02_rc_rr_cj_blsh_lr4e4_dpp02_epoch_6.pth \
#--deterministic
#CUDA_VISIBLE_DEVICES=0 python \
#/home/whua/code/ie_e2e/tools/train.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_trie/ephoie/local/ephoie_def_norot_gt_norot_200e_local_1062.py \
#--work-dir=/home/whua/logs/ie_e2e_log/temp_run \
#--load-from=/home/whua/logs/ie_e2e_weights/sroie_FORMAL_3L_screen_ft_default_dp02_rr_3l_lr2e4_bs4_epoch_570.pth
# --load-from=/home/whua/logs/ie_e2e_weights/sroie_FORMAL_3L_screen_ft_default_dp02_rr_3l_lr2e4_bs4_epoch_570.pth
# --load-from=/home/whua/logs/ie_e2e_weights/ocr_pretrain_eng_full_default_dp02_rc_rr_cj_blsh_lr4e4_dpp02_epoch_6.pth
# --load-from=/home/whua/logs/ie_e2e_weights/sroie_FORMAL_screen_ft_default_dp02_lr2e4_rr_bs4_epoch_300.pth \
# -------- tmp ---------
#CUDA_VISIBLE_DEVICES=0 python /home/whua/code/ie_e2e/tools/train.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_sdef_rr_nark_3l_rec1_TAM_INS_1e1_200e_cloud_tal.py \
#--work-dir=/home/whua/logs/ie_e2e_log/debug_run \
#--deterministic
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
#/path/to/code/ie_e2e/tools/train.py \
#/path/to/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_screen_serial_sdef_lncrf_rr_nark_rec1_glbar_200e_cloud_tal.py \
#--work-dir=/save/dir/path --launcher pytorch --gpus 4 \
#--load-from=/path/to/weights/sroie_FORMAL_screen_ft_default_dp02_lr2e4_rr_bs4_epoch_300.pth \
#--deterministic
