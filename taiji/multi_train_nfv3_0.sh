export WORKSPACE=/apdcephfs/share_887471/interns/v_willwhua
export WORKSPACE_SELF=/apdcephfs/share_887471/interns/v_willwhua
# 0520_ie_ar_nfv4_ft_default_dp02_lr1e4_bs4_0
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
${WORKSPACE}/code/ie_e2e/tools/train.py ${WORKSPACE}/code/ie_e2e/configs/vie_custom/e2e_ar_vie/nfv4/nfv4_default_dp02_lr1e4_cloud.py \
--work-dir=${WORKSPACE_SELF}/logs/ie_ar_vie_log_0516/nfv4_ft_default_dp02_lr1e4_bs4 --launcher pytorch --gpus 4 \
--load-from=/apdcephfs/share_887471/interns/v_willwhua/logs/ie_ar_ocr_log_0421/ocr_pretrain_eng_full_default_dp02_rc_rr_cj_blsh_lr4e4_dpp02/epoch_6.pth \
--deterministic