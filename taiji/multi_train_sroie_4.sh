export WORKSPACE=/apdcephfs/share_887471/common/whua
## FINETUNE_E2E_VIE_SGD
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
#${WORKSPACE}/code/ie_e2e/tools/train.py ${WORKSPACE}/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/sroie_ar_1280_4e4_ocr20_cloud.py \
#--work-dir=${WORKSPACE}/logs/ie_ar_e2e_log/sroie_finetune_ocr20_sgd --launcher pytorch --gpus 4 \
#--load-from=/apdcephfs/share_887471/common/whua/logs/ie_ar_e2e_log/ocr_pretrain_custom_1728/latest.pth
# FINETUNE_E2E_VIE_BOTH_NO_NODE
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
${WORKSPACE}/code/ie_e2e/tools/train.py ${WORKSPACE}/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/sroie_ar_1280_1e4_both_ocr80_kie80_nonode_adam_cloud.py \
--work-dir=${WORKSPACE}/logs/ie_ar_e2e_log/sroie_finetune_both_ocr80_kie80_nonode_adam_1e4 --launcher pytorch --gpus 4 \
--load-from=/apdcephfs/share_887471/common/whua/logs/ie_ar_e2e_log/ocr_pretrain_custom_eng_1280_adam/latest.pth