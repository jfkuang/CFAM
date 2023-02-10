#single train
#CUDA_VISIBLE_DEVICES=0 python /home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/cord/cord_baseline_ie_head_kvc_1280_200e_1062.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/test

#11.6
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10063 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_local_1062.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/VIES_GT --launcher pytorch --gpus 4 \
#--deterministic --seed 3407

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10069 ./tools/train.py \
./configs/vie_custom/e2e_ar_vie/v5/Ours.py \
--work-dir=./logs/ie_e2e_log/ours_self_attention --launcher pytorch --gpus 4 \
--deterministic --seed 3407

# CUDA_VISIBLE_DEVICES=0 python ./tools/train.py \
# ./configs/vie_custom/e2e_ar_vie/v5/Ours.py \
# --work-dir=./logs/ie_e2e_log/ours_self_attention

#single test
#CUDA_VISIBLE_DEVICES=0 python /home/jfkuang/code/ie_e2e/tools/test.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_local_1062.py \
#/home/jfkuang/logs/ie_e2e_log/ours_GT/epoch_10.pth \
#--eval hmean-iou  --show-dir /data2/jfkuang/logs/vis/test