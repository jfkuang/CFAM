#single test
#CUDA_VISIBLE_DEVICES=3 python /home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_local_1061.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/test

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10019 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_local_1061.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/learnable_as_entity_feature_noadd --launcher pytorch --gpus 4 \
#--deterministic --seed 3407


#CUDA_VISIBLE_DEVICES=1 python /home/jfkuang/code/ie_e2e/tools/test.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_local_1061.py \
#/home/jfkuang/logs/ie_e2e_log/epoch_160.pth \
#--eval hmean-iou

#trie
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10049 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_trie/v5/local/nfv5_3125_sdef_3l_200e_720_local_1061.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/trie_nfv5 --launcher pytorch --gpus 4 \
#--deterministic --seed 3407


#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10019 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_local_1061.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/learnable_as_entity_feature --launcher pytorch --gpus 4 \
#--deterministic --seed 3407

#11.20 self_attention
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10069 \
/home/jfkuang/code/ie_e2e/tools/train.py \
/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_local_1061.py \
--work-dir=/home/jfkuang/logs/ie_e2e_log/ours_self_attention --launcher pytorch --gpus 4 \
--deterministic --seed 3407