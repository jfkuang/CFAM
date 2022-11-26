export WORKSPACE=/apdcephfs/share_887471/interns/v_willwhua
export WORKSPACE_SELF=/apdcephfs/share_887471/interns/v_willwhua
# 0608_trie_ar_sroie_ft_sroie_default_rr_norot_bs4_0
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
${WORKSPACE}/code/ie_e2e/tools/train.py ${WORKSPACE}/code/ie_e2e/configs/vie_custom/e2e_trie/sroie/sroie_sc_sdef_norot_cloud.py \
--work-dir=${WORKSPACE_SELF}/logs/ie_trie_log/sroie_ft_sroie_trie_def_rr_norot_bs4 --launcher pytorch --gpus 4 \
--load-from=/apdcephfs/share_887471/interns/v_willwhua/logs/ie_ocr_weights/sroie_FORMAL_screen_ft_default_dp02_lr2e4_rr_bs4_epoch_300.pth \
--deterministic