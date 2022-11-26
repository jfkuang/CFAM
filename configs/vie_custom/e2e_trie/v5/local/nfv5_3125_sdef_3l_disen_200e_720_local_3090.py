#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ie_e2e 
@File    ：nfv5_3125_sdef_3l_disen_200e_720_local_3090.py
@IDE     ：PyCharm 
@Author  ：jfkuang
@Date    ：2022/11/8 19:21 
'''

from configs._base_.det_models.ocr_mask_rcnn_r50_fpn_ohem import model as det
# Be aware, the num_classes in ocr_mask_rcnn_r50_fpn_ohem_poly is set to 80, which is incorrect
# for ocr tasks.

_base_ = [
    '../../../_base_/vie_datasets/nfv5_3125_3090.py',
    '../../../_base_/e2e_pipelines/vie_manner_pipeline_aug_720.py'
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
        use_mask_align=True
    ),
    ext_head=dict(
        type='CustomTRIE',
        rec_cfg=dict(
            text_encoder_args=dict(
                type='none',
                input_dim=512,
            ),
            d_model=512,
            fusion_pe_args=dict(
                n_position=200,
            ),
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
            # dual layer no longer has the same meaning as in our method
            rec_dual_layer_num=1,
            gt_mode=True,
            # max_seq_len here includes both SOS and EOS
            max_seq_len=125,
            infor_context_module=dict(
                type='MultiModalContextModule',
                textual_embedding=dict(
                    type='NodeEmbedding',
                    dropout_ratio=0.1,
                    merge_type='Sum',
                    pos_embedding=dict(
                        type='PositionEmbedding2D',
                        max_position_embeddings=64,
                        embedding_dim=512,
                        width_embedding=False,
                        height_embedding=False,
                    ),
                    sentence_embedding=dict(
                        type='SentenceEmbeddingCNN',
                        embedding_dim=512,
                        kernel_sizes=[3, 5, 7, 9]
                    ),
                ),
                multimodal_fusion_module=dict(
                    type='MultiModalFusion',
                    merge_type='Weighted',
                    visual_dim=[512],
                    semantic_dim=[512],
                ),
                textual_relation_module=dict(
                    type='BertEncoder',
                    config=dict(
                        hidden_size=512,
                        num_hidden_layers=2,
                        num_attention_heads=16,
                        intermediate_size=512,  # 4 x hidden_size
                        hidden_act="gelu",
                        hidden_dropout_prob=0.1,
                        attention_probs_dropout_prob=0.1,
                        layer_norm_eps=1e-12,
                        output_attentions=False,
                        output_hidden_states=False,
                        is_decoder=False, )
                )
            ),
            crf_args=dict(
                disentangle=True,
                use_kie_loss=True,
                keep_kie_res=True,
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
                ins_lvl_mean=False
            ),
        ),
        loss=dict(
            type='MASTERTFLoss',
            ocr_ignore_index=2,
            kie_ignore_index=1,
            reduction='mean',
            rec_weights=10.0,
            kie_weights=10.0,
            crf_weights=10.0
        ),
        use_ins_mask=True,
        rec_only=False,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
        ),
    vis_cfg=dict(
        show_score=False,
        show_bbox=False,
        show_text=False,
        show_entity=False,
        dict_file='/data3/jfkuang/data/nfv5_3125/dict.json',
        class_file='/data3/jfkuang/data/nfv5_3125/class_list.json',
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
evaluation = dict(interval=10, metric='e2e-hmean')

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
load_from = "/data3/jfkuang/weights/ocr_pretrain_eng_full_default_dp02_rc_rr_cj_blsh_lr4e4_dpp02_epoch_6.pth"
resume_from = None
workflow = [('train', 1)]

find_unused_parameters = True
# fp16 = dict(loss_scale='dynamic')
