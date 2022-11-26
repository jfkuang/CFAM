#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/5 15:31
# @Author : WeiHua
from configs._base_.det_models.ocr_mask_rcnn_r50_fpn_ohem import model as det
# Be aware, the num_classes in ocr_mask_rcnn_r50_fpn_ohem_poly is set to 80, which is incorrect
# for ocr tasks.

_base_ = [
    '../_base_/ocr_datasets/synthtext_cloud.py',
    '../_base_/e2e_pipelines/ocr_pretrain_pipeline.py'
]

det_module = det.copy()
det_module['type'] = 'ReI_OCRMaskRCNN'
det_module['roi_head']['type'] = 'ReI_StandardRoIHead'

# ------------ model setting ------------
model = dict(
    type='TwoStageSpotter',
    det_head=det_module,
    neck=dict(
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=(14, 14), sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        spatial_embedding=dict(
            max_position_embeddings=128,
            embedding_dim=256,
            width_embedding=True,
            height_embedding=True),
        use_mask_align=True
    ),
    ext_head=dict(
        type='AsynReaderLocal',
        rec_cfg=dict(
            node_attn_cfg=dict(
                d_model=256,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1
            ),
            pred_cfg=dict(
                in_chans=256,
                embed_dim=384,
                depth=12,
                num_heads=6,
                mlp_ratio=4,
                qkv_bias=True,
                max_len=60,
                out_depth=2
            ),
            node_attn_layers=4,
            txt_classes=97,
            entity_classes=19,
            use_crf=False
        ),
        fusion_pe_args=dict(
            d_hid=256,
            n_position=200
        ),
        loss=dict(
            type='MultiStepLoss',
            ignore_index=-1,
            reduction='mean',
            rec_weights=1.0,
            kie_weights=4.0
        ),
        use_ins_mask=True,
        use_seq_mask=True,
        rec_only=True,
        ignore_val=-1,
        end_val=1,
        iter_num=2,
        iter_num_local=2,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None
        ),
    vis_cfg=dict(
        show_score=False,
        show_bbox=True,
        show_text=True,
        show_entity=False,
        dict_file='/apdcephfs/share_887471/common/whua/dataset/ie_e2e/ie_e2e_data/mm_format/table/dict.json',
        class_file=None
    ),
    pretrain=None,
    init_cfg=None,
    train_cfg=None,
    test_cfg=None,
    part_pretrain=None,
    #part_pretrain=dict(EXT='/apdcephfs/share_887471/common/whua/logs/ie_e2e_weights/vitstr_small_patch16_224.pth'),
    model_type='OCR')


train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

# ------------ data preparation ------------
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=4),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline))

# additional parameter of evaluation setting.
#   interval: the interval of evaluation.
#   metric: which is specified in dataset class.
evaluation = dict(interval=50, metric='e2e-hmean')

# ------------ optimizer & scheduler ------------
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.001,
    step=[16, 18])
total_epochs = 20


# ------------ run time & logging ------------
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook')
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = '/apdcephfs/share_887471/common/whua/logs/ie_e2e_log/ocr_pretrain/latest.pth'
workflow = [('train', 1)]

find_unused_parameters = True
# fp16 = dict(loss_scale='dynamic')


