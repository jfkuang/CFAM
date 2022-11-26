#before running:
# every config change code_root,eg: ../configs/vie_custom/e2e_ar_vie/v5/ldk/method/nfv5_3125_sdef_kvc_200e_720_method1.py
# data config change data_root, eg: configs/vie_custom/e2e_ar_vie/v5/ldk/_base_/nfv5_3125_ar_local.py

#NO.1 method 1
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10009 \
#train.py \
#../configs/vie_custom/e2e_ar_vie/v5/ldk/method/nfv5_3125_sdef_kvc_200e_720_method1.py \
#--work-dir=../logs/ie_e2e_log/method1 --launcher pytorch --gpus 4 \
#--deterministic --seed=3407
#
##NO.2 texture_feature encoder
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10013 \
#train.py \
#../configs/vie_custom/e2e_ar_vie/v5/ldk/encoder/nfv5_3125_sdef_encoder_texture_kvc_200e_720_local_1062.py \
#--work-dir=../logs/ie_e2e_log/texture_encoder --launcher pytorch --gpus 4 \
#--deterministic --seed=3407

#NO.3 vis_feature encoder
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10012 \
#train.py \
#../configs/vie_custom/e2e_ar_vie/v5/ldk/encoder/nfv5_3125_sdef_encoder_vis_kvc_200e_720_local_1062.py \
#--work-dir=../logs/ie_e2e_log/vis_encoder --launcher pytorch --gpus 4 \
#--deterministic --seed=3407


#NO.4 both embedding
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10017 \
#train.py \
#../configs/vie_custom/e2e_ar_vie/v5/ldk/embedding/nfv5_3125_sdef_enbedding_both_kvc_200e_720_local_1062.py \
#--work-dir=../logs/ie_e2e_log/both_embedding --launcher pytorch --gpus 4 \
#--deterministic --seed=3407


#NO.5 texture_feature embedding
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10016 \
#train.py \
#../configs/vie_custom/e2e_ar_vie/v5/ldk/embedding/nfv5_3125_sdef_embedding_texture_kvc_200e_720_local_1062.py \
#--work-dir=../logs/ie_e2e_log/texture_embedding --launcher pytorch --gpus 4 \
#--deterministic --seed=3407

#NO.6 vis_feature embedding
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10015 \
#train.py \
#../configs/vie_custom/e2e_ar_vie/v5/ldk/embedding/nfv5_3125_sdef_embedding_vis_kvc_200e_720_local_1062.py \
#--work-dir=../logs/ie_e2e_log/vis_embedding --launcher pytorch --gpus 4 \
#--deterministic --seed=3407

#NO.7 both encoder
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10014 \
#train.py \
#../configs/vie_custom/e2e_ar_vie/v5/ldk/encoder/nfv5_3125_sdef_encoder_both_kvc_200e_720_local_1062.py \
#--work-dir=../logs/ie_e2e_log/both_encoder --launcher pytorch --gpus 4 \
#--deterministic --seed=3407

#NO.8 learning rate
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10010 \
#train.py \
#../configs/vie_custom/e2e_ar_vie/v5/ldk/lr/nfv5_3125_sdef_lr3e4_kvc_200e_720_local_1062.py \
#--work-dir=../logs/ie_e2e_log/lr --launcher pytorch --gpus 4 \
#--deterministic --seed=3407


#NO.9 lr policy
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10011 \
#train.py \
#../configs/vie_custom/e2e_ar_vie/v5/ldk/lr/nfv5_3125_sdef_lr_policy_50_kvc_200e_720_local_1062.py \
#--work-dir=../logs/ie_e2e_log/lr_policy --launcher pytorch --gpus 4 \
#--deterministic --seed=3407







