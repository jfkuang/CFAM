export WORKSPACE=/apdcephfs/private_v_fisherwyu/code
export LOGDIR=/apdcephfs/share_887471/interns/v_fisherwyu/ie_e2e_log
# Custom English V1 Dataset Fuse Encode No-Det No-Node
# 0323_ie_ar_sroie_scratch_default_fv1bn_4l_nonode_bs6_0
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 \
${WORKSPACE}/code/ie_e2e/tools/train.py ${WORKSPACE}/code/ie_e2e/configs/vie_custom/e2e_ar_vie/sroie/sroie_ar_default_fusev1bn_4layer_nonode_shuffle_cloud_wwyu.py \
--work-dir=${LOGDIR}/test_run --launcher pytorch --gpus 3 \
--deterministic