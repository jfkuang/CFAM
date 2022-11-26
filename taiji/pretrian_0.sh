export WORKSPACE=/apdcephfs/private_v_fisherwyu/code
export LOGDIR=/apdcephfs/share_887471/interns/v_fisherwyu/ie_e2e_log
# Custom English V1 Dataset Fuse Encode No-Det No-Node
# 0321_ie_ar_pretrain_custom_eng_v1_fusev0_encode_mlp_nonode_nodet_adam_1e4_2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 \
${WORKSPACE}/code/ie_e2e/tools/train.py ${WORKSPACE}/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/custom_dataset/custom_eng_v1_encode_mlp_nonode_nodet_ar_1280_adam_cloud.py \
--work-dir=${LOGDIR}/ocr_pretrain_custom_eng_v1_fusev0_encode_mlp_nonode_nodet --launcher pytorch --gpus 8 \
--deterministic