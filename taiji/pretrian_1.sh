export WORKSPACE=/apdcephfs/share_887471/interns/v_willwhua
# Custom English V1 Dataset Fuse Encode No-Det No-Node
# 0526_ie_ar_pretrain_syn_chn_default_dp02_rc_rr_cj_blsh_lr4e4_dpp02_0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 \
${WORKSPACE}/code/ie_e2e/tools/train.py ${WORKSPACE}/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/custom_dataset/synth_chn_default_dp02_rc_rr_cj_blsh_lr4e4_dpp02_cloud.py \
--work-dir=${WORKSPACE}/logs/ie_ar_ocr_log_0514/ocr_pretrain_syn_chn_default_dp02_rc_rr_cj_blsh_lr4e4_dpp02 --launcher pytorch --gpus 8 \
--load-from=${WORKSPACE}/logs/ie_ar_ocr_log_0514/ocr_pretrain_syn_chn_default_dp02_rc_rr_cj_blsh_lr4e4_dpp02/latest.pth \
--deterministic