#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/12 17:11
# @Author : WeiHua

from .ar_reader import AutoRegReader
from mmocr.models.spotting.recognizers.old.ar_reader_v1 import AutoRegReaderV1
from mmocr.models.spotting.recognizers.old.ar_reader_idpdt import AutoRegReaderIDPDT
from .ar_reader_serial import AutoRegReaderSerial
from .ar_reader_serial_local_ie import AutoRegReaderSerialLocalIE
from .re_imple_trie.trie import CustomTRIE
from .ar_reader_nar_ie import AutoRegReaderNARIE
from .counters import CSRNetDecoder
from .ar_reader_nar_ie_0726 import AutoRegReaderNARIE0726
# from .rnn_attention_nar_ie import RNNRecNARIE
from .ar_reader_nar_ie_0726_kvc import AutoRegReaderNARIE0726_kvc
from .ar_reader_nar_ie_0726_kvc_decoder import AutoRegReaderNARIE0726_kvc_decoder
from .ar_reader_nar_ie_0726_kvc_head import AutoRegReaderNARIE0726_kvc_head

__all__ = [
    'AutoRegReader', 'AutoRegReaderV1',
    'AutoRegReaderIDPDT', 'AutoRegReaderSerial',
    'AutoRegReaderSerialLocalIE', 'CustomTRIE',
    'AutoRegReaderNARIE', 'CSRNetDecoder',
    'AutoRegReaderNARIE0726', 'AutoRegReaderNARIE0726_kvc',
    'AutoRegReaderNARIE0726_kvc_head', 'AutoRegReaderNARIE0726_kvc_decoder'
]
