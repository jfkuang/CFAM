export WORKSPACE=/apdcephfs/share_887471/common/whua
# FINETUNE_E2E_VIE
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
${WORKSPACE}/code/ie_e2e/tools/train.py ${WORKSPACE}/code/ie_e2e/configs/vie_custom/e2e_vie/sroie/sroie_mask_align_input_1280_adam_cloud.py \
--work-dir=${WORKSPACE}/logs/ie_e2e_log/sroie_finetune_ocr20_adam_1e4 --launcher pytorch --gpus 4 \
--load-from=/apdcephfs/share_887471/common/whua/logs/ie_e2e_log/ocr_pretrain_custom_1728/latest.pth