#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ie_e2e 
@File    ：sroie_ie_1032.py
@IDE     ：PyCharm 
@Author  ：jfkuang
@Date    ：2022/10/15 22:36 
'''
from configs._base_.det_models.ocr_mask_rcnn_r50_fpn_ohem import model as det
# Be aware, the num_classes in ocr_mask_rcnn_r50_fpn_ohem_poly is set to 80, which is incorrect
# for ocr tasks.

_base_ = [
    '../../../_base_/vie_datasets/sroie_screen_ar_local_1032.py',
    '../../../_base_/e2e_pipelines/vie_manner_pipeline_aug_rr_1280.py'
]

det_module = det.copy()
det_module['type'] = 'ReI_OCRMaskRCNN'
det_module['roi_head']['type'] = 'ReI_StandardRoIHead'
det_module['test_cfg'].update(sort_test=True)

# ------------ model setting ------------
model = dict(
    type='TwoStageSpotter',
    det_head=det_module,
    neck=dict(
        feature_fuser=dict(
            feat_num_in=4, in_channels=256, out_channels=512, sum_up=True,
        ),
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=(10, 40), sampling_ratio=0),
            out_channels=512,
            featmap_strides=[4, ]),
        spatial_embedding=dict(
            embedding_dim=512,
        ),
        use_mask_align=True
    ),
    ext_head=dict(
        type='AutoRegReaderNARIE0726_kvc',
        rec_cfg=dict(
            text_encoder_args=dict(
                type='none',
                input_dim=512,
            ),
            d_model=512,
            fusion_pe_args=dict(
                n_position=200,
            ),
            node_attn_cfg=dict(
                type='origin',
                use_layout_node=False,
                use_context_node=False,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
            ),
            node_attn_layers=4,
            rec_pe_args=dict(
                n_position=1000,
            ),
            feat_pe_args=dict(
                dropout=0.1,
                n_position=1000
            ),
            rec_decoder_args=dict(
                self_attn=dict(
                    headers=8,
                    d_model=512,
                    dropout=0.2),
                src_attn=dict(
                    headers=8,
                    d_model=512,
                    dropout=0.2),
                feed_forward=dict(
                    d_model=512,
                    d_ff=2024,
                    dropout=0.2),
                size=512,
                dropout=0.2),
            rec_layer_num=3,
            rec_dual_layer_num=1,
            # max_seq_len here includes both SOS and EOS
            max_seq_len=72,
            crf_args=dict(
                use_kie_loss=False,
                lstm_args=dict(
                    bilstm_kwargs=dict(
                        input_size=-1,
                        hidden_size=512,
                        num_layers=2,
                        dropout=0.1,
                        bidirectional=True,
                        batch_first=True
                    ),
                    mlp_kwargs=dict(
                        in_dim=-1,
                        out_dim=-1,
                        dropout=0.1
                    ),
                    apply_norm=True
                ),
                use_crf=True,
                ins_lvl_mean=False,
                use_both_modal=False,
                kv_catcher_args=dict(
                    type='OneStageContrast',
                    kvc_feat_type='E',
                    use_context_encoder=True,
                    use_texture_encoder=True,
                    use_context_embedding=False,
                    use_texture_embedding=False,
                    kvc_weights=10.0,
                    add_decoder=False,
                    kvc_type='matrix'
                )
            )
        ),
        loss=dict(
            type='MASTERTFLoss',
            ocr_ignore_index=2,
            kie_ignore_index=1,
            reduction='mean',
            rec_weights=1.0,
            kie_weights=10.0,
            crf_weights=10.0
        ),
        use_ins_mask=True,
        rec_only=False,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
        query_cfg=dict(
            query_type='OCR_ONLY',
            ocr_weight=0.5,
            kie_weight=0.5,
            forward_manner='PARALLEL'),
        ),
    vis_cfg=dict(
        show_score=False,
        show_bbox=True,
        show_text=True,
        show_entity=True,
        dict_file='/home/whua/dataset/ie_e2e_dataset/sroie/e2e_format/dict.json',
        class_file='/home/whua/dataset/ie_e2e_dataset/sroie/e2e_format//class_list.json',
        auto_reg=True
    ),
    pretrain=None,
    init_cfg=None,
    train_cfg=None,
    test_cfg=None,
    part_pretrain=None,
    model_type='VIE')
model['ext_head']['ocr_dict_file'] = model['vis_cfg']['dict_file']
model['ext_head']['kie_dict_file'] = model['vis_cfg']['class_file']


train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

# ------------ data preparation ------------
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2),
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
evaluation = dict(interval=10, metric='e2e-hmean-sroie')

# ------------ optimizer & scheduler ------------
# optimizer
optimizer = dict(type='Adam', lr=2e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[100, 150])
total_epochs = 200


# ------------ run time & logging ------------
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook')
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = '/apdcephfs/share_887471/common/whua/logs/ie_e2e_log/ocr_pretrain/latest.pth'
load_from = '/home/whua/logs/ie_e2e_weights/sroie_FORMAL_3L_screen_ft_default_dp02_rr_3l_lr2e4_bs4_epoch_570.pth'
resume_from = None
workflow = [('train', 1)]

find_unused_parameters = True
# fp16 = dict(loss_scale='dynamic')