dataset_type = 'VIEE2EDataset'
data_root = '/home/jfkuang/data/ephoie_e2e_format/e2e_format'
loader = dict(
    type='HardDiskLoader',
    repeat=1,
    parser=dict(
        type='CustomLineJsonParser',
        keys=['file_name', 'height', 'width', 'annotations'],
        optional_keys=['entity_dict']))
train = dict(
    type='VIEE2EDataset',
    ann_file='/home/jfkuang/data/ephoie_e2e_format/e2e_format/train.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='CustomLineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'],
            optional_keys=['entity_dict'])),
    dict_file='/home/jfkuang/data/ephoie_e2e_format/e2e_format/dict.json',
    img_prefix='/home/jfkuang/data/ephoie_e2e_format/e2e_format',
    pipeline=None,
    test_mode=False,
    class_file=
    '/home/jfkuang/data/ephoie_e2e_format/e2e_format/class_list.json',
    data_type='ocr',
    max_seq_len=80,
    order_type='shuffle',
    auto_reg=True,
    pre_parse_anno=True)
test = dict(
    type='VIEE2EDataset',
    ann_file='/home/jfkuang/data/ephoie_e2e_format/e2e_format/test.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='CustomLineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'],
            optional_keys=['entity_dict'])),
    dict_file='/home/jfkuang/data/ephoie_e2e_format/e2e_format/dict.json',
    img_prefix='/home/jfkuang/data/ephoie_e2e_format/e2e_format',
    pipeline=None,
    test_mode=True,
    class_file=
    '/home/jfkuang/data/ephoie_e2e_format/e2e_format/class_list.json',
    data_type='ocr',
    max_seq_len=80,
    order_type='origin',
    auto_reg=True)
train_list = [
    dict(
        type='VIEE2EDataset',
        ann_file='/home/jfkuang/data/ephoie_e2e_format/e2e_format/train.txt',
        loader=dict(
            type='HardDiskLoader',
            repeat=1,
            parser=dict(
                type='CustomLineJsonParser',
                keys=['file_name', 'height', 'width', 'annotations'],
                optional_keys=['entity_dict'])),
        dict_file='/home/jfkuang/data/ephoie_e2e_format/e2e_format/dict.json',
        img_prefix='/home/jfkuang/data/ephoie_e2e_format/e2e_format',
        pipeline=None,
        test_mode=False,
        class_file=
        '/home/jfkuang/data/ephoie_e2e_format/e2e_format/class_list.json',
        data_type='ocr',
        max_seq_len=80,
        order_type='shuffle',
        auto_reg=True,
        pre_parse_anno=True)
]
test_list = [
    dict(
        type='VIEE2EDataset',
        ann_file='/home/jfkuang/data/ephoie_e2e_format/e2e_format/test.txt',
        loader=dict(
            type='HardDiskLoader',
            repeat=1,
            parser=dict(
                type='CustomLineJsonParser',
                keys=['file_name', 'height', 'width', 'annotations'],
                optional_keys=['entity_dict'])),
        dict_file='/home/jfkuang/data/ephoie_e2e_format/e2e_format/dict.json',
        img_prefix='/home/jfkuang/data/ephoie_e2e_format/e2e_format',
        pipeline=None,
        test_mode=True,
        class_file=
        '/home/jfkuang/data/ephoie_e2e_format/e2e_format/class_list.json',
        data_type='ocr',
        max_seq_len=80,
        order_type='origin',
        auto_reg=True)
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
max_scale = 1280
min_scale = 1280
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='MaskOutInstance'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='CustomResize',
        img_scale=(1280, 1280),
        keep_ratio=True,
        multiscale_mode='range'),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='RandomRotate', angles=[-10, 10]),
    dict(type='BrightnessTransform', level=5, prob=0.5),
    dict(type='ContrastTransform', level=5, prob=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='E2EFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_polys'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='CustomResize',
        img_scale=(1280, 1280),
        keep_ratio=True,
        multiscale_mode='range'),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='E2EFormatBundle'),
    dict(type='Collect', keys=['img'])
]
det = dict(
    type='OCRMaskRCNN',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4],
            ratios=[0.03, 0.06, 0.1, 0.16, 0.25, 0.17, 0.44, 1.13, 2.9, 7.46],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='ReI_StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
                gpu_assign_thr=50),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.75),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='OHEMSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.45),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.3),
            max_per_img=100,
            mask_thr_binary=0.5),
        sort_test=True))
det_module = dict(
    type='ReI_OCRMaskRCNN',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4],
            ratios=[0.03, 0.06, 0.1, 0.16, 0.25, 0.17, 0.44, 1.13, 2.9, 7.46],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='ReI_StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
                gpu_assign_thr=50),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.75),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='OHEMSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.45),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.3),
            max_per_img=100,
            mask_thr_binary=0.5),
        sort_test=True))
model = dict(
    type='TwoStageSpotter',
    det_head=dict(
        type='ReI_OCRMaskRCNN',
        backbone=dict(
            type='mmdet.ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50'),
            norm_eval=True,
            style='pytorch'),
        neck=dict(
            type='mmdet.FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[4],
                ratios=[
                    0.03, 0.06, 0.1, 0.16, 0.25, 0.17, 0.44, 1.13, 2.9, 7.46
                ],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        roi_head=dict(
            type='ReI_StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            mask_head=dict(
                type='FCNMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                    gpu_assign_thr=50),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_across_levels=False,
                nms_pre=2000,
                nms_post=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.75),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='OHEMSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_across_levels=False,
                nms_pre=1000,
                nms_post=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.45),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.3),
                max_per_img=100,
                mask_thr_binary=0.5),
            sort_test=True)),
    neck=dict(
        feature_fuser=dict(
            feat_num_in=4, in_channels=256, out_channels=512, sum_up=True),
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign', output_size=(10, 40), sampling_ratio=0),
            out_channels=512,
            featmap_strides=[4]),
        spatial_embedding=dict(embedding_dim=512),
        use_mask_align=False),
    ext_head=dict(
        type='AutoRegReader',
        rec_cfg=dict(
            text_encoder_args=dict(type='none', input_dim=512),
            d_model=512,
            fusion_pe_args=dict(n_position=200),
            node_attn_cfg=dict(
                type='origin',
                use_layout_node=False,
                use_context_node=False,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1),
            node_attn_layers=4,
            rec_pe_args=dict(n_position=1000),
            feat_pe_args=dict(dropout=0.1, n_position=1000),
            rec_decoder_args=dict(
                self_attn=dict(headers=8, d_model=512, dropout=0.2),
                src_attn=dict(headers=8, d_model=512, dropout=0.2),
                feed_forward=dict(d_model=512, d_ff=2024, dropout=0.2),
                size=512,
                dropout=0.2),
            rec_layer_num=4,
            rec_dual_layer_num=1,
            max_seq_len=72,
            crf_args=dict()),
        loss=dict(
            type='MASTERTFLoss',
            ocr_ignore_index=2,
            kie_ignore_index=1,
            reduction='mean',
            rec_weights=10.0,
            kie_weights=10.0),
        use_ins_mask=False,
        rec_only=True,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
        query_cfg=dict(
            query_type='BOTH',
            ocr_weight=0.5,
            kie_weight=0.5,
            forward_manner='PARALLEL'),
        ocr_dict_file=
        '/home/jfkuang/data/ephoie_e2e_format/e2e_format/dict.json',
        kie_dict_file=
        '/home/jfkuang/data/ephoie_e2e_format/e2e_format/class_list.json'),
    vis_cfg=dict(
        show_score=False,
        show_bbox=True,
        show_text=True,
        show_entity=False,
        dict_file='/home/jfkuang/data/ephoie_e2e_format/e2e_format/dict.json',
        class_file=
        '/home/jfkuang/data/ephoie_e2e_format/e2e_format/class_list.json',
        auto_reg=True),
    pretrain=None,
    init_cfg=None,
    train_cfg=dict(use_det_res=True, max_ins_num_per_img=80),
    test_cfg=None,
    part_pretrain=None,
    model_type='OCR')
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2),
    train=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='VIEE2EDataset',
                ann_file=
                '/home/jfkuang/data/ephoie_e2e_format/e2e_format/train.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='CustomLineJsonParser',
                        keys=['file_name', 'height', 'width', 'annotations'],
                        optional_keys=['entity_dict'])),
                dict_file=
                '/home/jfkuang/data/ephoie_e2e_format/e2e_format/dict.json',
                img_prefix='/home/jfkuang/data/ephoie_e2e_format/e2e_format',
                pipeline=None,
                test_mode=False,
                class_file=
                '/home/jfkuang/data/ephoie_e2e_format/e2e_format/class_list.json',
                data_type='ocr',
                max_seq_len=80,
                order_type='shuffle',
                auto_reg=True,
                pre_parse_anno=True)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(type='MaskOutInstance'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='CustomResize',
                img_scale=(1280, 1280),
                keep_ratio=True,
                multiscale_mode='range'),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='RandomRotate', angles=[-10, 10]),
            dict(type='BrightnessTransform', level=5, prob=0.5),
            dict(type='ContrastTransform', level=5, prob=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='E2EFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_polys'])
        ]),
    val=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='VIEE2EDataset',
                ann_file=
                '/home/jfkuang/data/ephoie_e2e_format/e2e_format/test.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='CustomLineJsonParser',
                        keys=['file_name', 'height', 'width', 'annotations'],
                        optional_keys=['entity_dict'])),
                dict_file=
                '/home/jfkuang/data/ephoie_e2e_format/e2e_format/dict.json',
                img_prefix='/home/jfkuang/data/ephoie_e2e_format/e2e_format',
                pipeline=None,
                test_mode=True,
                class_file=
                '/home/jfkuang/data/ephoie_e2e_format/e2e_format/class_list.json',
                data_type='ocr',
                max_seq_len=80,
                order_type='origin',
                auto_reg=True)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='CustomResize',
                img_scale=(1280, 1280),
                keep_ratio=True,
                multiscale_mode='range'),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='E2EFormatBundle'),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='VIEE2EDataset',
                ann_file=
                '/home/jfkuang/data/ephoie_e2e_format/e2e_format/test.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='CustomLineJsonParser',
                        keys=['file_name', 'height', 'width', 'annotations'],
                        optional_keys=['entity_dict'])),
                dict_file=
                '/home/jfkuang/data/ephoie_e2e_format/e2e_format/dict.json',
                img_prefix='/home/jfkuang/data/ephoie_e2e_format/e2e_format',
                pipeline=None,
                test_mode=True,
                class_file=
                '/home/jfkuang/data/ephoie_e2e_format/e2e_format/class_list.json',
                data_type='ocr',
                max_seq_len=80,
                order_type='origin',
                auto_reg=True)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='CustomResize',
                img_scale=(1280, 1280),
                keep_ratio=True,
                multiscale_mode='range'),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='E2EFormatBundle'),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=20, metric='e2e-hmean-sroie')
optimizer = dict(type='Adam', lr=0.0002)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[200, 400])
total_epochs = 400
checkpoint_config = dict(interval=20)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/jfkuang/code/ie_e2e/pretrained_models/epoch_25.pth'
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
work_dir = '/home/jfkuang/logs/ie_e2e_log/0713_train_pos_nms75'
gpu_ids = range(0, 2)
