#CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10211 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_3090.py \
#--work-dir=/data2/jfkuang/logs/ie_e2e_log/feature --launcher pytorch --gpus 4 \
#--deterministic --seed 3407

#python use_gpu.py --size 13000 --gpus 4 --interval 0.01

#ground-truth
#CUDA_VISIBLE_DEVICES=7 python /home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_3090.py \
#--work-dir=/data2/jfkuang/logs/ie_e2e_log/test
#CUDA_VISIBLE_DEVICES=6,7  python -m torch.distributed.launch --nproc_per_node=2 --master_port=10067 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_3090.py \
#--work-dir=/data3/jfkuang/logs/ie_e2e_log/VIES_GT --launcher pytorch --gpus 2 \
#--deterministic --seed 3407
#CUDA_VISIBLE_DEVICES=6,7  python -m torch.distributed.launch --nproc_per_node=2 --master_port=10066 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_3090.py \
#--work-dir=/data3/jfkuang/logs/ie_e2e_log/ours_GT --launcher pytorch --gpus 2 \
#--deterministic --seed 3407

#single test
#CUDA_VISIBLE_DEVICES=6 python /home/jfkuang/code/ie_e2e/tools/test.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_3090.py \
#/data3/jfkuang/logs/ie_e2e_log/vies_sroie_600epoch/epoch_600.pth \
#--eval hmean-iou --show-dir /data3/jfkuang/vis_weights_vies/vis_no_text_green/

#CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node=2 --master_port=10037 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_3090.py \
#--work-dir=/data3/jfkuang/logs/ie_e2e_log/complete_method_noadd --launcher pytorch --gpus 2 \
#--deterministic --seed 3407

CUDA_VISIBLE_DEVICES=2,3  python -m torch.distributed.launch --nproc_per_node=2 --master_port=10096 \
/home/jfkuang/code/ie_e2e/tools/train.py \
/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_3090.py \
--work-dir=/data3/jfkuang/logs/ie_e2e_log/baseline_no_encoder_no_CFAM --launcher pytorch --gpus 2 \
--deterministic --seed 3407