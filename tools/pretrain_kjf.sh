#DDP train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=10011 \
/home/jfkuang/code/ie_e2e/tools/train.py \
/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/custom_dataset/synth_chn_default_dp02_rc_lr2e4_dpp02_1803_30epoch_kjf.py \
--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_pretrain_chn_bs8_960_higher200_2e4 --launcher pytorch --gpus 8 \
--deterministic --resume-from=/home/jfkuang/logs/ie_e2e_log/ephoie_pretrain_chn_bs8_960_higher200_2e4/latest.pth

#train
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/custom_dataset/synth_chn_default_dp02_rc_rr_cj_blsh_lr4e4_dpp02_cloud_kjf.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_pretrain_chn_bs8_480_lower100  --gpus 8

#single try
#CUDA_VISIBLE_DEVICES=0 python /home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/custom_dataset/synth_chn_default_dp02_rc_rr_cj_blsh_lr4e4_dpp02_cloud_kjf.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_pretrain_chn_bs8_480_lower100  --gpus 1 \
#--deterministic