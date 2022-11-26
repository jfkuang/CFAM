#before running:
# every config change code_root,eg: ../configs/vie_custom/e2e_ar_vie/v5/ldk/method/nfv5_3125_sdef_kvc_200e_720_method1.py
# data config change data_root, eg: configs/vie_custom/e2e_ar_vie/v5/ldk/_base_/nfv5_3125_ar_local.py


#No.1 self-attention
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=11001 \
#train.py \
#../configs/vie_custom/e2e_ar_vie/v5/ldk/test/mfv5_3125_sdef_kvc_self_attention_200e_720.py \
#--work-dir=../logs/ie_e2e_log/self_attention --launcher pytorch --gpus 4 \
#--deterministic --seed=3407

#No.2 cross-attention query:entity_feature
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=11002 \
#train.py \
#../configs/vie_custom/e2e_ar_vie/v5/ldk/test/nfv5_3125_sdef_kvc_cross_attention_200e_720.py \
#--work-dir=../logs/ie_e2e_log/cross_attention --launcher pytorch --gpus 4 \
#--deterministic --seed=3407

#No.3 cross-attention2  query:instance_feature
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=11005 \
#train.py \
#../configs/vie_custom/e2e_ar_vie/v5/ldk/test/nfv5_3125_sdef_kvc_cross_attention2_200e_720.py \
#--work-dir=../logs/ie_e2e_log/cross_attention2 --launcher pytorch --gpus 4 \
#--deterministic --seed=3407

#No.4 clip
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=11003 \
#train.py \
#../configs/vie_custom/e2e_ar_vie/v5/ldk/test/nfv5_3125_sdef_kvc_clip_200e_720.py \
#--work-dir=../logs/ie_e2e_log/clip --launcher pytorch --gpus 4 \
#--deterministic --seed=3407

#No.5 feature
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=11004 \
#train.py \
#../configs/vie_custom/e2e_ar_vie/v5/ldk/test/nfv5_3125_sdef_kvc_none_200e_720.py \
#--work-dir=../logs/ie_e2e_log/feature --launcher pytorch --gpus 4 \
#--deterministic --seed=3407