export WORKSPACE=/apdcephfs/private_v_fisherwyu/code
export LOGDIR=/apdcephfs/share_887471/interns/v_fisherwyu/ie_e2e_log
# Custom English V1 Dataset Fuse Encode No-Det No-Node
# 0327_ie_ar_ephoie_finetune_default_ep10_fv1_4l_nonode_aug_0
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 \
${WORKSPACE}/code/ie_e2e/tools/train.py ${WORKSPACE}/code/ie_e2e/configs/vie_custom/e2e_ar_vie/ephoie/ephoie_ar_default_fv1_4l_nonode_shuffle_aug_cloud.py \
--work-dir=${LOGDIR}/ephoie_finetune_ep10_default_fv1_4l_aug_nonode_bs6 --launcher pytorch --gpus 2 \
--load-from=${LOGDIR}/ocr_pretrain_custom_chn_v1_fv1_4l_nonode_nodet/epoch_10.pth \
--deterministic