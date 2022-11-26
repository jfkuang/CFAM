#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/12/28 22:48
# @Author : WeiHua

"""
AR Default setting:
in size: 1280
kie_weight: 4.0
forward_manner: parallel
query_type: both
query_ocr_weight: 0.5
query_kie_weight: 0.5
instance_sort: shuffle
train_from: scratch


WORK_DIR_CONFIG:
Pytorch 1.7.1
torchvision 0.8.2
mmcv: 1.4.2
mmdet: 2.19.1
mmocr: 0.4.0

"""

"""
Todo:
    To speed up, the det results won't be used in REC for now. Build the whole end-to-end model as soon as possible.
    1. Prepare the setting for constructing an end-to-end model
        1.1 Check if current dataset & dataloader is available, and the required format of dataset (Done)
        1.2 Read an implementation of end-to-end model based on MMOCR (Done)
        1.3 Read ViTSTR code & paper (Done)
        1.4 Read TRIE code, to determine the way of connecting DET, REC and IE. (Done)
            1.4.1 TRIE-GT
            1.4.2 TRIE
        1.5 Mark the part that needed to be modified / add / deleted & training protocol (Done)
    2. Finish the code and test -- This week ~ 1.12
        3.1 Build OCR part -> Mask R-CNN + ViTSTR
            3.1.1 implement dataset, similar to "IcdarDataset" -> (Done)
                a. finish code
                b. add visualization
            3.1.2 build end-to-end ocr model -> 1.15 DDL
                prep. build dataset with required format. (Done)
                a. build without REC and IE head, run it. (Done)
                -> verify if whole system works:
                    1) if label generating is ok (through visualizer)
                    2) the output of detection is as expected (visualize the
                    output of detection, both box and segmentation)
                b. add REC, run it. (Done)
                    1) try to add detection proposal
                    2) use gt bbox & mask, along with MMOCR's default Roi_Align,
                    try to implement Mask_Align later
                    3) add ViTSTR as REC branch
                    4) add part_load function to load weights
                c. add KIE, run it. (Most Done)
                    1) optimize the code to reduce memory cost & fix the bug about instance_mask
                        1. AMP in MMOCR
                        2. view & reshape & transpose & permute
                    2) implement AsyncReaderLocal & ViTSTRLocal, the Async is only exists in out_block of ViTSTR,
                    the global_iter=2, local_iter=2.
                    3) add KIE branch, similar to REC branch
                    4) design CRF-Layer (Not yet)
                d. add Inference (Done)
                    1) finish simple_test of end2end class, especially detection part.
                        -> both rescale results and after-process results are required
                    2) add "show_result" function in end2end class, and visualize the detection results
                e. Unfinished
                    1) add CRF-Layer
    3. Experiment Plan
        3.1 E2E-VIE with ocr pretrain
        3.2 E2E-VIE with different kie weights
        3.3 E2E-OCR with different ROI size & ocr pretrain

MMOCR:
[1] https://yimiandai.me/post/blog-content/
"""


"""
expectation:
    1. Higher performance on SROIE, FUNSD
    2. Higher performance on Our dataset.
    3. Evidence that can proves the IE and REC branches can help each other.
    4. Report inference time.

TRIE's code is based on OpenMM-Lab, which is a good news.

Text Detection: Mask R-CNN
    -> pre-trained model of MM-OCR
Text Recognition: ViTSTR -> "Vision Transformer for Fast and Efficient Scene Text Recognition"
    -> pre-trained model of ViTSTR, build this part with the same procedure of ViTSTR (based on timm)
Info Extraction: Similar structure of ViTSTR, e.g. Transformer Encoder.
    -> considering using the same Transformer architecture of ViTSTR (share weights)
    -> the output layer is designed respectively, for ie, use CRF layer additionally.

OCR characters in ViTSTR:
    import string
    [GO] + [SPACE] + string.printable[:-6] # [GO] acts as start and padding, [SPACE] acts as end.

Dataset: Similar to KIEDataset, but the format should be the same as Mask-R-CNN
    1. Spatial Feature -> based on bounding box and polygon, ignore rotate boxes for now.

Model construction: Follow the implementation of TRIE.
    # 1. Directly use DAVAR-Lab's BaseEndToEnd and TwoStageEndToEnd as our end-to-end model class.
    1. Based on OCRMaskRCNN, rewrite the forward_train & forward_test. Take forward_train as example:
        a. Add additional REC and IE head after roi-head to achieve end-to-end VIE.
        b. Use ground-truth for recognition. Besides,
            -> add an alternative to choose if adding prediction. Similar to ABCNet's implementation,
            (see adet/modeling/roi_heads/text_head.py -> line 194)
            first match each prediction proposal with all ground-truth, then choose the best matched one
            and assign the ground-truth label to this prediction proposal.
        c. Add a sample module to sub-sample the proposal and ground-truth. (For OCR only)
        -> Later, it is not necessary. Sub-sample is only required for detection, maybe add it
        later, same as ABCNet or Mask Text-Spotter. (to be decided)

Train protocol:
    1. For detection, use the default setting.
    2. For end-to-end OCR, add ground-truth to proposals of detection.
    3. For end-to-end IE, use ground-truth as detection result, and use the prediction results of REC and IE as
    the inner input for the other module.
    4. Later, switch to fill-with-ground-truth mode -> to be decided.
    5. Load weights for mask r-cnn (DET) and ViTSTR (REC) respectively, then end-to-end pretrain OCR on NF-V1.
    Current we use DET and REC pretrained model respectively, but in official experiments, we shall only pretrain the
    whole model in synth-text rather than train them separately.


Alert:
    1. In order to load weights for different parts of the model while remain the stem code, add pre_train arg to model,
    then the model can choose to load from pretrain when init.
    2. When re-implementation an module in OpenMM, remember to override its forward_train and forward_test. Besides, do
    not call train_step or val_step to perform train or test, since the input and output of these function are defined
    in its father class. In addition, the output format should follow the father class's implementation of train_step
    and val_step, otherwise you'll have to re-write these functions.
    -> General Pipeline: 
    Train:
        train_step -> call forward -> forward_train
    Eval:
        val_step -> call forward -> calculate val loss -> call EvalHook -> forward_test -> simple_test / aug_test
        (If there exists multiple augmentations, then it call aug_test, otherwise simple_test)
        (EvalHook -> mmocr/apis/train.py)
        '''
            eval_hook = DistEvalHook if distributed else EvalHook
            runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
        '''
        In EvalHook, it will call "single_gpu_test(runner.model, self.dataloader, show=False)"
        In single_gpu_test, rescale will be set to True, so the prediction will be resize to origin scale.
    3. When training OCR, there is an augmentation that combine proposals and gts to REC, but only gts when training VIE.
    4. For evaluation, gt is loaded in dataset.evaluate, then the metric is calculated.
    5. Polygons and boxes for REC & KIE branch are not sorted ! (not like PICK)
"""

### config setting ###
loader = dict(
    type='HardDiskLoader',
    repeat=1,
    parser=dict(
        type='LineJsonParser',
        keys=['file_name', 'height', 'width', 'annotations']))
"""
Dataset Format: (similar to WileReceipt)
└── Nutrition Facts
  ├── class_list.json
  ├── dict.json
  ├── image_files
  ├── test.txt
  └── train.txt

For test.txt & train.txt:
    Each line is a dict, includes [file_name, height, width, annotations, entity_dict (optional)].
    "annotations" corresponding to a list, each of which is a dict and contains keys:
    {
        "polygon": polygon, [x, y, x, y ...], expected to have the same number of vertex
        "text": transcript,
        "entity": entity tag in token-level, which is IOB tag format, e.g.,
            [entity_of_token_1, entity_of_token_2, ...]
    },
    "entity_dict" indicates a map from entity key to entity value. If it is provided, then it will
     directly be used to evaluate the KIE results. It is formatted as below:
    {
        "entity_key": "entity_val"
    }
    
    

For dict.json & class_list.json:
    a list of tokens, e.g., character for "dict.json" and entity class for "class_list.json".

"""


"""
Pipeline
before through pipeline, contains:
{
    'img_info' : {
                    'filename',
                    'height',
                    'width'   
                 }
    'ann_info' : {
                    'masks': polygons
                    'rboxes'
                    'bboxes'
                    'ori_texts'
                    'ori_entities': optional
                    'ann_type'
                    'labels': {
                                'instances': label for each box/polygon
                                'texts'
                                'entities': optional
                              } 
                 }
    'img_prefix': prefix of image path.
    'gt_polys': same as ['ann_info']['masks'], which is polygons of text instances
    'bbox_fields': ['gt_polys']
    'mask_fields': []
    'seg_fields': []
    'ori_texts': transcripts of texts
    'filename': full image file path
    'ori_filename': relative image file path
    'img': dummy input for image, seems no use.
}

1. LoadImageFromFile -> Load image data
-> It may exists bug, check the implementation 'DavarLoadImageFromFile' in TRIE.
    Update: 'filename', 'ori_filename', 'img', 'img_shape', 'ori_shape', 'img_fields' = ['img']
2. LoadAnnotations -> Convert annotation to the format model required
    with_bbox=True -> require ['ann_info']['bboxes']
        'gt_bboxes' = ['ann_info']['bboxes']
        Update 'gt_bboxes_ignore' and add key 'gt_bboxes_ignore' to 'bbox_fields' if exists.
        Add key 'gt_bboxes' to 'bbox_fields'
    with_label=Ture ->
        'gt_labels' = ['ann_info']['labels']
    with_mask=True -> convert 'masks' to masks
        'gt_masks', add key 'gt_masks' to 'mask_fields'
    with_seg=False
3. CustomResize -> Resize image, mask, polygon, box
    _resize_img -> resize img, which is specified by ['img_fields']
    _resize_bboxes -> resize bbbox & polys, which is specified by ['bbox_fields']
    _resize_cbboxes -> resize character box, which is specified by ['cbbox_fields']
    _resize_masks -> resize mask, which is specified by ['mask_fields']
    _resize_seg -> resize segmentation, which is specified by ['seg_fields']
4. RandomFlip -> randomly flip image, which will not be performed here.
5. CustomVisualizer -> Visualize annotations to see if there is a problem.
    visualize_box -> visualize based on 'bbox_fields'
    visualize_mask -> visualize based on 'mask_fields'
6. Normalize -> Normalize image based on 'img_fields'
7. Pad -> Padding to specific size or divisible
    -> 'img_fields', 'mask_fields', 'seg_fields' 
8. E2EFormatBundle ->
    -> convert img to float type if required
    -> convert img to Tensor, dim = C, H, W
    -> convert 'gt_bboxes', 'gt_bboxes_ignore' to DataContainer
    -> convert annotations in 'gt_labels' to DataContainer
9. Collect
    -> select 'img', 'gt_bboxes', 'gt_masks', 'gt_labels' and 'img_metas'(by default), then return
"""

"""
forward_train of end-to-end model:
    1. detection by Mask R-CNN
        1.1 return loss, feature maps (list[Tensor]), detection results (optional)
    2. feature extraction & roi-align (or roi-masking) -> visual feature
        -> aligned feature of different instances of different images
        -> pad to same shape and stack together for batch operation
        -> spatial feature: N, C, L
    3. prepare spatial feature
    4. read by AsynReader
        4.1 multi-modal fusion
        4.2 Recognition
        4.3 MLP-layer & CRF-layer
"""

"""
Expr Plan:
1. Modify instance-level refinement, the mask will only mask pad tokens, rather than the to-predict token in query.
    -- expr running
    --> ins_refine1 -> not mask current, ins_refine -> mask current
2. Add node-feature to query, too.
3. During sequence decoding, from B*N, L, C to B*L, N, C,
B*N, L, C -> B*N, N, L, C -> 

-> fulfill results under ocr=kie=8.0
-> figure out expr plan

pre-train weight path:
custom_english_v1: /apdcephfs/share_887471/common/whua/logs/ie_ar_e2e_log/ocr_pretrain_custom_1728/latest.pth

custom_english_v2: /apdcephfs/share_887471/common/whua/logs/ie_ar_e2e_log/ocr_pretrain_custom_eng_1280_adam/latest.pth

custom_eng_v1: /apdcephfs/share_887471/common/whua/logs/ie_ar_e2e_log/ocr_pretrain_custom_eng_v1_rec10_nonode_nodet_1280_adam

custom_english_v2_gca: /apdcephfs/share_887471/common/whua/logs/ie_ar_e2e_log/ocr_pretrain_custom_eng_gca_1280_adam/latest.pth

synth_text: /apdcephfs/share_887471/common/whua/logs/ie_ar_e2e_log/ocr_pretrain_st_1280/latest.pth

"""

"""
todo:
DDL 3.11:
    1. get current best results
    2. finish global modeling module (both layout and context)
    3. finish the design of interaction
    4. finish the design of encoder for sequence decoding
    
-> Quit validate, write down some detailed setting for later validation (if time and resource are available)
1. update expr results (Done)
    Global Modeling:
    1. verify if context info works
    
    1. Find optimal setting of Layout Info 
    2. Find optimal position to use layout info -> input query or kie query 
    3. Find optimal setting of Context Info -> add to query or kie query,
        concat or add & weather to use it at last -> 3.10, 3.11
    Interaction:
    1. design of interaction -> 3.11
    Encoder of Sequence decoding:
    1. design of this encoder -> later
2. improve OCR results -> data augmentation & strong encoder & train strategy
3. run pre-train with current setting (freeze det-head & increase augmentation) -> prior 2
-> after 2 & 3, run on cloud
4. build strong encoder
5. design interaction mechanism
6. add a pseudo task for pretrain, to replace the KIE branch -> pseudo task design
7. Read paper !

-> TODAY !
    
Expr_Done:
1. Ablation study on layout info, for scratch, this module works (with around 1~2 points boost)
2. Ablation study on context info, for scratch, this module works (with around 1~2 points boost)


3.7~3.13
1. 提升模型OCR性能 (Boost Baseline)
	1.1 加入Master's Encoder --> Running
	1.2 增加数据增强 (仅针对OCR端到端训练) --> Linjie Tang
	1.3 训练策略的修改：预训练之后，先在对应数据集上finetune OCR再finetune端到端KIE (从而可以在finetune OCR的过程中引入上述数据增强) ->
	1.4 BS=4的结果似乎优于BS=6，目前正在验证

2. 确定Global Modeling的有效性：(Scratch if GPU not available, otherwise finetune)
	2.1 证明Layout模块的有效性 -> verified under scratch
	2.2 证明Context模块的有效性 -> verified under scratch
	2.3 证明Layout + Context模块能同时起作用

3. prepare branch interaction module

4. think about pesudo task for OCR pretrain

Expr running:
1033:
1. No-node, with text encoder, shuffle, batch_size=4
2. no_node, shuffle, batch_size=4
3. cat_l2k_last, shuffle, batch_size=4

1032:
1. context_to_kie, shuffle, batch_size=4
2. cat_sort_l2k_last, shuffle, batch_size=4
3. cat_l2k_last, context_to_kie, shuffle, batch_size=4



"""