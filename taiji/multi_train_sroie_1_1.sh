export WORKSPACE=/apdcephfs/share_887471/interns/v_willwhua
export WORKSPACE_SELF=/apdcephfs/share_887471/interns/v_willwhua
# 0609_ie_ar_sroie_sc_ft_sroie_default_rr_localie_reconly_narkie_300e_bs4_0
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
${WORKSPACE}/code/ie_e2e/tools/train.py ${WORKSPACE}/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/sroie_screen_serial_sdef_lncrf_rr_localie_reconly_narkie_300e_cloud.py \
--work-dir=${WORKSPACE_SELF}/logs/ie_ar_vie_log_0516/sroie_sc_ft_serial_sroie_def_rr_crf_localie_reconly_narkie_300e_bs4 --launcher pytorch --gpus 4 \
--load-from=/apdcephfs/share_887471/interns/v_willwhua/logs/ie_ocr_weights/sroie_FORMAL_screen_ft_default_dp02_lr2e4_rr_bs4_epoch_300.pth \
--deterministic