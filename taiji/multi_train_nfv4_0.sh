export WORKSPACE=/apdcephfs/share_887471/interns/v_willwhua
# FINETUNE NO-NODE FUSE-V0(NO-SUM-UP) BS=6
# 0327_ie_ar_nfv4_default_finetune_ep10_fv1_4l_nonode_aug_shuffle_1
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
${WORKSPACE}/code/ie_e2e/tools/train.py ${WORKSPACE}/code/ie_e2e/configs/vie_custom/e2e_ar_vie/nfv4/nfv4_ar_default_fv1_4l_nonode_shuffle_aug_cloud.py \
--work-dir=${WORKSPACE}/logs/ie_ar_e2e_log_2/nfv4_finetune_ep10_default_fv1_4l_nonode_aug_bs4 --launcher pytorch --gpus 4 \
--load-from=/apdcephfs/share_887471/interns/v_willwhua/logs/ie_ar_e2e_log_1/ocr_pretrain_eng_full_fusev1_4layer_nonode_nodet/epoch_10.pth \
--deterministic