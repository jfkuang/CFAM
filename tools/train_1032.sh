#CUDA_VISIBLE_DEVICES=0,1,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10018 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/local/sroie_ie_1032.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/no_head_kvc_sroie --launcher pytorch --gpus 4 \
#--deterministic --seed=3407   #--seed=3407,1364371869

#single test
#CUDA_VISIBLE_DEVICES=7 python train.py \
#../configs/vie_custom/e2e_trie/cord/cord_trie_1280_200e_1062.py \
#--work-dir=../logs/ie_e2e_log/test

#CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10012 \
#train.py \
#../configs/vie_custom/e2e_ar_vie/cord/cord_baseline_1280_200e_1032.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/kvc_method1 --launcher pytorch --gpus 2 \
#--deterministic --seed=3407

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10019 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_ie_nodecoder_3l_200e_720_local_1032.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ours_classification --launcher pytorch --gpus 4 \
#--deterministic --seed=3407

#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10019 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_local_1032.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/vie_nfv5 --launcher pytorch --gpus 4 \
#--deterministic --seed=3407

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10039 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_local_1032.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/baseline_noCFAM --launcher pytorch --gpus 4 \
#--deterministic --seed=3407

CUDA_VISIBLE_DEVICES=2 python /home/jfkuang/code/ie_e2e/tools/test.py \
/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_local_1032.py \
 /home/jfkuang/logs/ie_e2e_log/vie_nfv5/epoch_200.pth \
--eval hmean-iou