export WORKSPACE=/apdcephfs/share_887471/interns/v_willwhua
export WORKSPACE_SELF=/apdcephfs/share_887471/interns/v_willwhua
# FINETUNE NO-NODE FUSE-V0(NO-SUM-UP) BS=6
## 0318_ie_ar_sroie_ocr_default_finetune_ep5_encode_fusev0_nonode_shuffle_0
#CUDA_VISIBLE_DEVICES=0 python ${WORKSPACE}/code/ie_e2e/tools/train.py \
#${WORKSPACE}/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/sroie_screen_idpdt_nsq_default_dp02_lr2e4_rr_cloud.py \
#--work-dir=${WORKSPACE_SELF}/logs/ie_ar_e2e_log_0331/single_test
CUDA_VISIBLE_DEVICES=0 python ${WORKSPACE}/code/ie_e2e/tools/test.py \
${WORKSPACE}/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/sroie_screen_idpdt_nsq_default_dp02_lr2e4_rr_cloud.py \
/apdcephfs/share_887471/interns/v_willwhua/logs/ie_ar_vie_log_0516/sroie_screen_ft_nsq_default_dp02_lr2e4_rr_bs4/epoch_30.pth --eval 'h-mean-sroie' \
--show-dir=${WORKSPACE_SELF}/logs/ie_ar_e2e_log_0331/single_test