#single test
#CUDA_VISIBLE_DEVICES=6 python /home/jfkuang/code/ie_e2e/tools/test.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/ephoie/ephoie_default_dp02_lr2e4_noalign_add_det_epoch600_pretrain_1032_kjf.py \
#/home/jfkuang/logs/ie_e2e_log/ephoie_baseline_noalign_epoch600_adddet_pretrain25_1280/epoch_600.pth \
#--eval hmean-iou --show-dir /home/jfkuang/logs/vis/test_new

#vies
CUDA_VISIBLE_DEVICES=5 python /home/jfkuang/code/ie_e2e/tools/test.py \
/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_kvc_ie_200e.py \
/data3/jfkuang/logs/ie_e2e_log/vies_sroie_600epoch/epoch_600.pth \
--eval hmean-iou-sroie  --show-dir /data3/jfkuang/vis_sroie/vis_text_red/

#trie
#CUDA_VISIBLE_DEVICES=5 python /home/jfkuang/code/ie_e2e/tools/test.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_trie/v5/local/nfv5_3125_sdef_3l_disen_200e_720_local_3090.py \
#/data3/jfkuang/vis_weights_trie/epoch_170.pth \
#--eval hmean-iou  --show-dir /data3/jfkuang/vis_weights_trie/vis_no_text_green/

#ours
#CUDA_VISIBLE_DEVICES=5 python /home/jfkuang/code/ie_e2e/tools/test.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_3090_vis.py \
#/data3/jfkuang/vis_weights_ours/epoch_160.pth \
#--eval hmean-iou  --show-dir /data3/jfkuang/vis_weights_ours/vis_no_text_red/