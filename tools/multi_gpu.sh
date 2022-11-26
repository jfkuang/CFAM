#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
#--nproc_per_node=2 --master_port=8888 \
#/home/whua/code/ie_e2e/tools/train.py '/home/whua/code/ie_e2e/configs/vie_custom/e2e_vie/default_config_e2e.py' \
#--work-dir='/mnt/whua/ie_e2e_log/test_run'
#CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh /home/whua/code/ie_e2e/configs/vie_custom/e2e_vie/default_config_e2e.py /mnt/whua/ie_e2e_log/test_run 4
#CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 /apdcephfs/share_887471/common/whua/code/ie_e2e/tools/dist_train.sh \
#/apdcephfs/share_887471/common/whua/code/ie_e2e/configs/vie_custom/e2e_vie/asyncreader_local_config_e2e.py \
#/apdcephfs/share_887471/common/whua/logs/ie_e2e_log/formal_train 4
export WORKSPACE=/apdcephfs/share_887471/common/whua
echo ${WORKSPACE}
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
${WORKSPACE}/code/ie_e2e/tools/train.py ${WORKSPACE}/code/ie_e2e/configs/vie_custom/e2e_vie/asyncreader_local_config_e2e.py \
--work-dir=${WORKSPACE}/logs/ie_e2e_log/formal_train --launcher pytorch --gpus 4