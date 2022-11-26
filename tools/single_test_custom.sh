#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=6 python /home/whua/project/ie_e2e/tools/test.py \
#/home/whua/project/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/nfv3_ar_1280_1e4_both_ocr20_adam_local_1032.py \
#/home/whua/logs/ie_weights/sroie_finetune_nonode_epoch_330.pth --eval 'h-mean' \
#--show-dir='/home/whua/logs/ie_e2e_log/sroie_finetune_nonode_vis'
#CUDA_VISIBLE_DEVICES=0 python /home/whua/code/ie_e2e/tools/test.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/sroie/local/sroie_default_dp02_randcrop_cj_local_9999.py \
#/data/whua/ie_ocr_log/sroie_default_dp02_randcrop_cj_bs4/epoch_600.pth --eval 'h-mean-sroie'
# 9999
# multi-card ~ visualization is not available
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 \
#/home/whua/code/ie_e2e/tools/test.py --launcher pytorch \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_screen_serial_sdef_lncrf_gm_c2ks_local_9999.py \
#/data/whua/ie_ocr_log/single_test/latest.pth --eval 'h-mean-sroie'
#epoch_names=('epoch_600' 'epoch_570' 'epoch_540' 'epoch_510' 'epoch_480' 'epoch_450' 'epoch_420' 'epoch_390' 'epoch_360' 'epoch_330' 'epoch_300' 'epoch_270' 'epoch_240')
#for (( i = 0; i < 13; i++ )); do
#    echo ${epoch_names[i]}
#    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 \
#    /home/whua/code/ie_e2e/tools/test.py --launcher pytorch \
#    /home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_screen_default_dp02_lr2e4_rr_local_9999.py \
#    /data/whua/ie_e2e_log_0516/sroie_screen_ft_default_dp02_lr2e4_rr_bs4/${epoch_names[i]}.pth --eval 'h-mean-sroie'
#done
# single-card
#CUDA_VISIBLE_DEVICES=0 python /home/whua/code/ie_e2e/tools/test.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/sroie/local/sroie_screen_default_dp02_lr2e4_rr_1803.py \
#/home/whua/logs/ie_ocr_weights/sroie_screen_ft_default_dp02_lr2e4_rr_bs4_epoch_390.pth --eval 'h-mean-sroie'
#CUDA_VISIBLE_DEVICES=0 python \
#/home/whua/code/ie_e2e/tools/test.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/sroie/local/sroie_default_dp02_lr2e4_rr_cj_bl_local_1061.py \
#/home/whua/logs/ie_ocr_log/sroie_ft_default_dp02_lr2e4_rr_cj_bl_bs4/epoch_510.pth --eval 'h-mean-sroie' \
#--show-dir='/home/whua/logs/ie_ocr_log/vis_sroie_ft_default_dp02_lr2e4_rr_cj_bl_bs4_epoch_510'
# 1061
#CUDA_VISIBLE_DEVICES=0 python /home/whua/code/ie_e2e/tools/test.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_screen_v1_serial_default_dp02_lr2e4_rr_local_1061.py \
#/home/whua/logs/ie_e2e_log/sroie_screen_ft_v1_default_dp02_lr2e4_rr_bs4/epoch_480.pth --eval 'h-mean-sroie' \
#--show-dir='/home/whua/logs/vis_sroie_screen_ft_v1_default_dp02_lr2e4_rr_bs4'
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
#/home/whua/code/ie_e2e/tools/test.py --launcher pytorch \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_screen_v1_serial_default_dp02_lr2e4_rr_local_1061.py \
#/home/whua/logs/ie_e2e_log/sroie_screen_ft_v1_default_dp02_lr2e4_rr_bs4/epoch_480.pth --eval 'h-mean-sroie'
#epoch_names=('epoch_600' 'epoch_570' 'epoch_540' 'epoch_510' 'epoch_480' 'epoch_450' 'epoch_420')
#for (( i = 0; i < 7; i++ )); do
#    echo ${epoch_names[i]}
#    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
#    /home/whua/code/ie_e2e/tools/test.py --launcher pytorch \
#    /home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/sroie/local/sroie_screen_default_dp02_lr2e4_rr_local_1061.py \
#    /home/whua/logs/ie_weights/sroie_screen_ft_default_dp02_rr_lr2e4_bs4/${epoch_names[i]}.pth --eval 'h-mean-sroie'
#done
# 1803
# multi-gpu
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=28500 \
#/home/whua/code/ie_e2e/tools/test.py --launcher pytorch \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_sdef_rr_nark_3l_rec1_cuskvc_KR_1STG_200e_local_1803.py \
#/home/whua/logs/ie_e2e_log/sroie_sdef_SEED_rr_nark_3l_rec1_cuskvc_KR_1STG_200e_bs4/latest.pth --eval 'h-mean-sroie'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
/home/whua/code/ie_e2e/tools/test.py --launcher pytorch \
/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_nark_3l_seq_wkie_200e_720_local_1803.py \
/home/whua/logs/ie_e2e_log/v5_3125_sdef_SEED_noinplace_nark_3l_seq_wkie_200e_720_local_200e_720_bs4/epoch_110.pth --eval 'e2e-hmean'
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=27500 \
#/home/whua/code/ie_e2e/tools/test.py --launcher pytorch \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_screen_idpdt_sq_default_dp02_lr1e4_rr_local_1803.py \
#/home/whua/logs/ie_ocr_weights/epoch_60.pth --eval 'h-mean-sroie'
# single-gpu
#CUDA_VISIBLE_DEVICES=7 python /home/whua/code/ie_e2e/tools/test.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_nark_3l_cuskvc_KR_1STG_200e_720_local_1803.py \
#/home/whua/logs/ie_e2e_log/v5_3125_sdef_SEED_nark_3l_cuskvc_KR_1STG_200e_720_bs4/latest.pth --eval 'h-mean'
# 1032
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 --master_port=28500 \
#/home/whua/project/ie_e2e/tools/test.py --launcher pytorch \
#/home/whua/project/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_screen_serial_sdef_lncrf_rr_localie_reconly_gtocr_local_1032.py \
#/home/whua/logs/ie_weights/sroie_sc_ft_serial_sroie_def_rr_lncrf_localie_reconly_bs4_epoch_270.pth --eval 'h-mean-sroie'
#CUDA_VISIBLE_DEVICES=6 python /home/whua/project/ie_e2e/tools/test.py \
#/home/whua/project/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_sdef_rr_nark_3l_rec1_cuskvc_KR_1h_1STG_200e_local_1032.py \
#/home/whua/logs/ie_e2e_log/sroie_sdef_SEED_rr_nark_3l_rec1_cuskvc_KR_1h_1STG_200e_bs4/latest.pth --eval 'h-mean-sroie'
#--show-dir='/home/whua/logs/ie_e2e_log/vis_v5_kv_catcher'
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=28500 \
#/home/whua/project/ie_e2e/tools/test.py --launcher pytorch \
#/home/whua/project/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_screen_serial_sdef_lncrf_rr_nark_3l_rec1_seq_200e_local_1032.py \
#/home/whua/logs/ie_weights/epoch_160.pth --eval 'h-mean-sroie'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 --master_port=26500 \
/home/whua/project/ie_e2e/tools/test.py --launcher pytorch \
/home/whua/project/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_sdef_rr_nark_3l_rec1_200e_local_1032.py \
/home/whua/logs/epoch_80.pth --eval 'h-mean-sroie'
#--show-dir='/home/whua/logs/ie_e2e_log/vis_non_disen'
# 1062
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
#/home/whua/code/ie_e2e/tools/test.py --launcher pytorch \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/sroie/local/sroie_screen_default_dp02_lr2e4_rr_local_1062.py \
#/home/whua/logs/ie_e2e_weights/sroie_FORMAL_screen_ft_default_dp02_lr2e4_rr_bs4_epoch_300.pth --eval 'h-mean-sroie'
#CUDA_VISIBLE_DEVICES=0 python /home/whua/code/ie_e2e/tools/test.py \
#/home/whua/code/ie_e2e/configs/vie_custom/e2e_trie/ephoie/local/ephoie_def_norot_gt_norot_200e_local_1062.py \
#/home/whua/logs/ie_e2e_log/trie_ephoie_def_norot_GT_200e_bs4/latest.pth --eval 'h-mean' \
#--show-dir='/home/whua/logs/trie_e2e/vis_trie_ephoie_def_norot_GT_200e_bs4'#
