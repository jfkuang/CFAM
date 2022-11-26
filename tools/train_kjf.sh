#load pretrained model
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10000 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/ephoie/local/ephoie_screen_serial_sdef_lncrf_rr_nark_3l_rec1_200e_local_1803.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/0729ephoie_600epoch --launcher pytorch --gpus 4 \
#--load-from=/home/jfkuang/code/ie_e2e/pretrain_models/epoch_10.pth \
#--deterministic

#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10001 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/ephoie/local/ephoie_screen_serial_sdef_lncrf_rr_nark_3l_rec1_200e_local_1803.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/0729ephoie_400epoch --launcher pytorch --gpus 4 \
#--load-from=/home/jfkuang/code/ie_e2e/pretrain_models/epoch_10.pth \
#--deterministic

#not load model 1803
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10010 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/ephoie/local/ephoie_default_serial_dp02_lr2e4_lncrf_norot_1803_kjf.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_ft_eng_def_se_lncrf_norot_dp02_lr2e4_bs4 --launcher pytorch --gpus 4 \
#--deterministic

#not load model 1032
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10010 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/ephoie/local/ephoie_default_serial_dp02_lr2e4_lncrf_1032_kjf.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_baseline --launcher pytorch --gpus 4 \
#--deterministic

#not load model 1061 baseline DDP
#noalign
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10010 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/ephoie/ephoie_default_dp02_lr2e4_noalign_1061_kjf.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_baseline_noalign --launcher pytorch --gpus 4 \
#--deterministic
#rr
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10010 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/ephoie/ephoie_default_dp02_lr2e4_rr_1061_kjf.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_baseline_rr --launcher pytorch --gpus 4 \
#--deterministic --resume-from=/home/jfkuang/logs/ie_e2e_log/ephoie_baseline_rr/latest.pth

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10012 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/ephoie/ephoie_default_dp02_lr2e4_noalign_epoch400_adddet_1061_kjf.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_baseline_noalign_adddet_400_add_gt --launcher pytorch --gpus 4 \
#--deterministic

#adddet+400epoch+transform3+polygon
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10012 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/ephoie/ephoie_default_dp02_lr2e4_noalign_epoch400_adddet_1061_kjf.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_baseline_noalign_adddet_400epoch_transform3_polygon --launcher pytorch --gpus 4 \
#--deterministic


#adddet+400epoch+noalign+roi_size(10,40)->(20,40)
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10012 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/ephoie/ephoie_default_dp02_lr2e4_noalign_epoch400_adddet_1061_kjf.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_baseline_noalign_adddet_roi2040 --launcher pytorch --gpus 4 \
#--deterministic

#not load model 1032 baseline+add det DDP+bs4
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10010 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/ephoie/ephoie_default_dp02_lr2e4_rr_1032_kjf.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_baseline_add_det --launcher pytorch --gpus 4 \
#--deterministic --resume-from=/home/jfkuang/logs/ie_e2e_log/ephoie_baseline_add_det/latest.pth

#add_det epoch400
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10010 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/ephoie/ephoie_default_dp02_lr2e4_epoch400_pretrain_1032_kjf.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_baseline_add_det_400epoch_pretrain25_1280_add_gt --launcher pytorch --gpus 4 \
#--deterministic

#kie encoderkie
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10010 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/ephoie/local/ephoie_serial_sdef_lncrf_rr_nark_enckie_1032.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_baseline_kie_encoder_600epoch_1280 --launcher pytorch --gpus 4 \
#--deterministic

#kie nark
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10010 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/ephoie/local/ephoie_serial_sdef_lncrf_rr_reconly_1032.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_baseline_kie_nark_600epoch_1280 --launcher pytorch --gpus 4 \
#--deterministic

#abi
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10010 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/ephoie/local/ephoie_serial_sdef_lncrf_rr_reconly_narkie_1032.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_baseline_kie_abi_600epoch_1280 --launcher pytorch --gpus 4 \
#--deterministic


#not load model 1032 baseline+rotate15_bs2 DDP
#CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10012 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/ephoie/ephoie_default_dp02_lr2e4_rr_nodet_rotate_1032_kjf.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_baseline_rotate5_bs2 --launcher pytorch --gpus 2 \
#--deterministic

#roi
#CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10012 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/ephoie/ephoie_default_dp02_lr2e4_roi_1032_kjf.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_baseline_roi_bs2 --launcher pytorch --gpus 2 \
#--deterministic

#noalign+add_det+pretrain600+1280
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10015 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/ephoie/ephoie_default_dp02_lr2e4_noalign_add_det_epoch600_pretrain_1032_kjf.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/0713_train_iou0.4_nms75 --launcher pytorch --gpus 4 \
#--deterministic

#vis
#CUDA_VISIBLE_DEVICES=6 python /home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_ocr_pretrain/ephoie/ephoie_default_dp02_lr2e4_noalign_1032_kjf.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/ephoie_baseline_noalign_test


#kvc_test
#CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10012 \
#/home/jfkuang/code/ie_e2e/tools/train.py \
#/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_local_1061.py \
#--work-dir=/home/jfkuang/logs/ie_e2e_log/0925_kvc --launcher pytorch --gpus 2 \
#--deterministic --seed 1364371869  #--seed=3407

CUDA_VISIBLE_DEVICES=3 python /home/jfkuang/code/ie_e2e/tools/train.py \
/home/jfkuang/code/ie_e2e/configs/vie_custom/e2e_ar_vie/v5/local/nfv5_3125_sdef_rnn_kvc_200e_720_local_1061.py \
--work-dir=/home/jfkuang/logs/ie_e2e_log/kvc0926