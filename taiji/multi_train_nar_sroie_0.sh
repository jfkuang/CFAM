export WORKSPACE=/apdcephfs/share_887471/common/whua
# FINETUNE_OCR_ONLY
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
${WORKSPACE}/code/ie_e2e/tools/train.py ${WORKSPACE}/code/ie_e2e/configs/vie_custom/e2e_ocr_pretrain/sroie/sroie_ocr_pretrain_adam_1280.py \
--work-dir=${WORKSPACE}/logs/ie_e2e_log/sroie_finetune_ocr_only_ocr20_adam --launcher pytorch --gpus 4 \
--resume-from=${WORKSPACE}/logs/ie_e2e_log/sroie_finetune_ocr_only_ocr20_adam/latest.pth
#--load-from=/apdcephfs/share_887471/common/whua/logs/ie_e2e_log/ocr_pretrain_custom_1728/latest.pth