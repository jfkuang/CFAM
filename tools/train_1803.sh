#CUDA_VISIBLE_DEVICES=2 python /home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_local_1803_vis.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/test

#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10011 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_ie_1803.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ours_another_seed --launcher pytorch --gpus 4 \
#--deterministic --seed 1364371869

#single test + vis
#CUDA_VISIBLE_DEVICES=3 python /home/jfkuang/code/ie_e2e/tools/test.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_local_1803_vis.py \
#/home/jfkuang/logs/ie_e2e_log/vies_kvc_nfv5/epoch_180.pth \
#--eval hmean-iou  --show-dir /home/jfkuang/logs/vis/vies_kvc_nfv5

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10019 \
/home/jfkuang/code/ie_e2e/tools/train.py \
/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/cord/cord_baseline_1280_200e_1803.py \
--work-dir=/home/jfkuang/logs/ie_e2e_log/ours_cord_600e_new_weights --launcher pytorch --gpus 4 \
--deterministic --seed 3407
