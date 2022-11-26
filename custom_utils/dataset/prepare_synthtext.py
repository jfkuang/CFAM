#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/27 22:40
# @Author : WeiHua

# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 6/28/2019 4:06 PM

# Multi-process crop synthtext and save it to lmdb or images/text file
import json
import shutil
from typing import *
import sys
import os
from itertools import chain
import math
import re
import logging
from multiprocessing import Queue, Pool, Process, Manager
from pathlib import Path
import argparse

import cv2
import numpy as np
from loguru import logger
import scipy.io as sio
import lmdb
import ipdb
from tqdm import tqdm

logger.remove(0)
logger.add('errors.log', level=logging.DEBUG)
logger.add(sys.stdout, level=logging.INFO)

QUEUE_SIZE = 50000
WORKERS = 8
LMDB_WRITE_BATCH = 5000

# Reference: https://github.com/wenwenyu/MASTER-pytorch/blob/main/data_utils/crop_synthtext.py
def crop_box_worker(args):
    '''
    crop synthtext by word bounding box, and put cropped data into queue
    '''
    image_name, txt, boxes, queue = args
    cropped_indx = 0

    # Get image name
    # print('IMAGE : {}'.format(image_name))

    # get transcript
    txt = [re.split(' \n|\n |\n| ', t.strip()) for t in txt]
    txt = list(chain(*txt))
    txt = [t for t in txt if len(t) > 0]

    # Open image
    # img = Image.open(image_name)
    img = cv2.imread(image_name, cv2.IMREAD_COLOR)
    img_height, img_width, _ = img.shape

    # Validation
    if len(np.shape(boxes)) == 2:
        wordBBlen = 1
    else:
        wordBBlen = boxes.shape[-1]

    if wordBBlen == len(txt):
        # Crop image and save
        for word_indx in range(len(txt)):
            if len(np.shape(boxes)) == 2:  # only one word (2,4)
                wordBB = boxes
            else:  # many words (2,4,num_words)
                wordBB = boxes[:, :, word_indx]

            if np.shape(wordBB) != (2, 4):
                err_log = 'malformed box index: {}\t{}\t{}\n'.format(image_name, txt[word_indx], wordBB)
                logger.debug(err_log)
                continue

            pts1 = np.float32([[wordBB[0][0], wordBB[1][0]],
                               [wordBB[0][3], wordBB[1][3]],
                               [wordBB[0][1], wordBB[1][1]],
                               [wordBB[0][2], wordBB[1][2]]])
            height = math.sqrt((wordBB[0][0] - wordBB[0][3]) ** 2 + (wordBB[1][0] - wordBB[1][3]) ** 2)
            width = math.sqrt((wordBB[0][0] - wordBB[0][1]) ** 2 + (wordBB[1][0] - wordBB[1][1]) ** 2)

            # Coord validation check
            if (height * width) <= 0:
                err_log = 'empty file : {}\t{}\t{}\n'.format(image_name, txt[word_indx], wordBB)
                logger.debug(err_log)
                continue
            elif (height * width) > (img_height * img_width):
                err_log = 'too big box : {}\t{}\t{}\n'.format(image_name, txt[word_indx], wordBB)
                logger.debug(err_log)
                continue
            else:
                valid = True
                for i in range(2):
                    for j in range(4):
                        if wordBB[i][j] < 0 or wordBB[i][j] > img.shape[1 - i]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    err_log = 'invalid coord : {}\t{}\t{}\t{}\t{}\n'.format(
                        image_name, txt[word_indx], wordBB, (width, height), (img_width, img_height))
                    logger.debug(err_log)
                    continue

            pts2 = np.float32([[0, 0],
                               [0, height],
                               [width, 0],
                               [width, height]])

            M = cv2.getPerspectiveTransform(pts1, pts2)
            img_cropped = cv2.warpPerspective(img, M, (int(width), int(height)))

            cropped_dir_name = image_name.split('/')[-2]
            cropped_file_name = "{}_{}_{}.jpg".format(cropped_indx,
                                                      image_name.split('/')[-1][:-len('.jpg')], word_indx)
            cropped_indx += 1
            data = dict(cropped_dir_name=cropped_dir_name,
                        filename=cropped_file_name,
                        transcript=txt[word_indx],
                        image=img_cropped)
            queue.put(data)

    else:
        err_log = 'word_box_mismatch : {}\t{}\t{}\n'.format(image_name,
                                                            txt,
                                                            boxes)
        logger.write(err_log)


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def lmdb_writer(lmdb_path: str, queue: Queue):
    env = lmdb.open(lmdb_path, map_size=1099511627776)
    buffer = {}
    counter = 0
    while True:
        data = queue.get()
        if data != 'Done':
            counter += 1
            img_cropped = data['image']
            img_cropped = cv2.imencode('.jpg', img_cropped)[1]
            buffer['image-{}'.format(counter)] = img_cropped.tobytes()
            buffer['transcript-{}'.format(counter)] = data['transcript'].encode()

            if counter % LMDB_WRITE_BATCH == 0 and counter != 0:
                writeCache(env, buffer)
                logger.info('{} done.'.format(counter))
                buffer = {}
        else:
            buffer['nSamples'] = str(counter).encode()
            writeCache(env, buffer)
            logger.info('Finished. Total {}'.format(counter))
            break


def images_with_gt_file_writer(images_path: str, gt_file: str, queue: Queue):
    gtfile = os.path.join(images_path, gt_file)
    counter = 0
    while True:
        data = queue.get()
        if data != 'Done':
            cropped_dir_name = data['cropped_dir_name']
            filename = data['filename']
            transcript = data['transcript']
            img_cropped = data['image']
            cropped_dir = os.path.join(images_path, cropped_dir_name)
            if not os.path.exists(cropped_dir):
                os.mkdir(cropped_dir)
            cropped_file_name = os.path.join(cropped_dir, filename)
            cv2.imwrite(cropped_file_name, img_cropped)
            with open(gtfile, 'a+', encoding='utf-8', ) as gt_f:
                gt_f.write('%s,%s\n' % (os.path.join(cropped_dir_name, filename), transcript))

            counter += 1

            if counter % LMDB_WRITE_BATCH == 0 and counter != 0:
                logger.info('{} done.'.format(counter))
        else:
            logger.info('Finished. Total {}'.format(counter))
            break


def lmdb_and_images_with_gt_file_writer(lmdb_path: str, images_path: str, gt_file: str, queue: Queue):
    env = lmdb.open(lmdb_path, map_size=1099511627776)
    gtfile = os.path.join(images_path, gt_file)

    buffer = {}
    counter = 0
    while True:
        data = queue.get()
        if data != 'Done':
            counter += 1
            img_cropped = data['image']
            transcript = data['transcript']
            img_cropped_buf = cv2.imencode('.jpg', img_cropped)[1]
            buffer['image-{}'.format(counter)] = img_cropped_buf.tobytes()
            buffer['transcript-{}'.format(counter)] = transcript.encode()

            # write to images and gt file
            cropped_dir_name = data['cropped_dir_name']
            filename = data['filename']
            cropped_dir = os.path.join(images_path, cropped_dir_name)
            if not os.path.exists(cropped_dir):
                os.mkdir(cropped_dir)
            cropped_file_name = os.path.join(cropped_dir, filename)
            cv2.imwrite(cropped_file_name, img_cropped)
            with open(gtfile, 'a+', encoding='utf-8', ) as gt_f:
                gt_f.write('%s,%s\n' % (os.path.join(cropped_dir_name, filename), transcript))

            # write to lmdb
            if counter % LMDB_WRITE_BATCH == 0 and counter != 0:
                writeCache(env, buffer)
                logger.info('{} done.'.format(counter))
                buffer = {}
        else:
            buffer['nSamples'] = str(counter).encode()
            writeCache(env, buffer)
            logger.info('Finished. Total {}'.format(counter))
            break


def synthtext_reader(synthtext_folder: str, queue: Queue, pool: Pool):
    synthtext_folder = Path(synthtext_folder)
    logger.info('Loading gt.mat ...')
    mat_contents = sio.loadmat(synthtext_folder.joinpath('gt.mat'))
    logger.info('Loading finish.')

    image_names = mat_contents['imnames'][0]
    # crop synthtext for every image, and put it into queue
    pool.map(crop_box_worker, iter([(synthtext_folder.joinpath(item[0]).absolute().as_posix(),
                                     mat_contents['txt'][0][index],
                                     mat_contents['wordBB'][0][index],
                                     queue)
                                    for index, item in enumerate(image_names[:])]))

    # for index, item in enumerate(image_names):
    #     crop_box_worker((synthtext_folder.joinpath('imgs/{}'.format(item[0])).absolute(),
    #                           mat_contents['txt'][0][index],
    #                           mat_contents['wordBB'][0][index],
    #                           queue))


def main(args):
    if not Path(args.synthtext_folder).exists():
        logger.error('synthtext_folder does not exist!')
        raise FileNotFoundError

    manager = Manager()
    queue = manager.Queue(maxsize=QUEUE_SIZE)

    # config data writer parallel process, read cropped data from queue, then save it to lmdb or images/txt file
    if args.data_format == 'lmdb':
        writer_process = Process(target=lmdb_writer, name='lmdb writer', args=(args.lmdb_path, queue), daemon=True)
    elif args.data_format == 'images_with_gt_file':
        Path(args.images_folder).mkdir(parents=True, exist_ok=True)
        writer_process = Process(target=images_with_gt_file_writer, name='images_with_gt_file writer',
                                 args=(args.images_folder, args.gt_file, queue), daemon=True)
    else:
        Path(args.images_folder).mkdir(parents=True, exist_ok=True)
        writer_process = Process(target=lmdb_and_images_with_gt_file_writer,
                                 name='lmdb_and_images_with_gt_file_writer writer',
                                 args=(args.lmdb_path, args.images_folder, args.gt_file, queue), daemon=True)
    writer_process.start()

    logger.info('{} writer is started with PID: {}'.format(args.data_format, writer_process.pid))

    # config synthtext data reader jobs
    pool = Pool(processes=WORKERS, maxtasksperchild=10000)
    try:
        logger.info('Start cropping...')
        # crop synthtext, and put cropped data into queue
        synthtext_reader(args.synthtext_folder, queue, pool)
        queue.put('Done')
        pool.close()
        pool.join()

        writer_process.join()
        writer_process.close()
        logger.info('End cropping.')
    except KeyboardInterrupt:
        logger.info('Terminated by Ctrl+C.')
        pool.terminate()
        pool.join()

def worker_st(synthtext_folder, saver_path, queue: Queue):
    logger_batch = 5000
    counter = 0
    while True:
        data = queue.get()
        if data != 'Done':
            _img_file = data['_img_file']
            box_ann = data['box_ann']
            txt_ann = data['txt_ann']
            # get transcript
            txt = [re.split(' \n|\n |\n| ', t.strip()) for t in txt_ann]
            txt = list(chain(*txt))
            txt = [t for t in txt if len(t) > 0]

            # Open image
            # img = Image.open(image_name)
            img_file = synthtext_folder.joinpath(_img_file[0]).absolute().as_posix()
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            img_height, img_width, _ = img.shape

            # Validation
            if len(np.shape(box_ann)) == 2:
                wordBBlen = 1
            else:
                wordBBlen = box_ann.shape[-1]

            assert wordBBlen == len(txt)
            out_info = dict(
                file_name=_img_file[0],
                height=img_height,
                width=img_width,
                annotations=list()
            )
            for idx in range(wordBBlen):
                if wordBBlen == 1:
                    cur_box = box_ann
                else:
                    cur_box = box_ann[:, :, idx]
                assert cur_box.shape == (2, 4)
                pts1 = np.float32([[cur_box[0][0], cur_box[1][0]],
                                   [cur_box[0][1], cur_box[1][1]],
                                   [cur_box[0][2], cur_box[1][2]],
                                   [cur_box[0][3], cur_box[1][3]]])
                instance = dict(
                    polygon=pts1.reshape(-1).tolist(),
                    text=txt[idx]
                )
                out_info['annotations'].append(instance)
            out_str = json.dumps(out_info)
            with open(saver_path, 'a+', encoding='utf-8') as f:
                f.write(out_str+'\n')
            counter += 1
            if counter % logger_batch == 0 and counter > 0:
                logger.info(f"{counter} done.")
        else:
            logger.info('Finished. Total {}'.format(counter))
            break


def prepare_synthtext(src_folder, out_folder):
    # src_ocr_dict = '/apdcephfs/share_887471/common/whua/dataset/ie_e2e/ie_e2e_data/mm_format/table/dict.json'
    # with open(src_ocr_dict, 'r', encoding='utf-8') as f:
    #     ocr_list = json.load(f)
    ext_ocr_list = []
    # loading synth-text gts
    # src_folder = '/apdcephfs/share_887471/common/ocr_benchmark/benchmark/SynthText'
    synthtext_folder = Path(src_folder)
    logger.info('Loading gt.mat ...')
    mat_contents = sio.loadmat(synthtext_folder.joinpath('gt.mat'))
    logger.info('Loading finish.')
    img_files = mat_contents['imnames'][0]
    box_anns = mat_contents['wordBB'][0]
    txt_anns = mat_contents['txt'][0]

    print("Converting to e2e_ie format...")
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.mkdir(out_folder)
    saver_path = os.path.join(out_folder, 'train.txt')

    manager = Manager()
    queue = manager.Queue(maxsize=QUEUE_SIZE)
    writer_process = Process(target=worker_st, name='synthtext writer',
                             args=(synthtext_folder, saver_path, queue,), daemon=True)
    writer_process.start()

    # config synthtext data reader jobs
    pool = Pool(processes=WORKERS, maxtasksperchild=10000)
    try:
        logger.info('Start converting...')
        # crop synthtext, and put cropped data into queue
        for _img_file, box_ann, txt_ann in tqdm(zip(img_files, box_anns, txt_anns)):
            data = dict(
                _img_file=_img_file,
                box_ann=box_ann,
                txt_ann=txt_ann
            )
            queue.put(data)
        queue.put('Done')
        pool.close()
        pool.join()

        writer_process.join()
        writer_process.close()
        logger.info('End converting.')
    except KeyboardInterrupt:
        logger.info('Terminated by Ctrl+C.')
        pool.terminate()
        pool.join()

    # for _img_file, box_ann, txt_ann in tqdm(zip(img_files, box_anns, txt_anns)):
    #     # get transcript
    #     txt = [re.split(' \n|\n |\n| ', t.strip()) for t in txt_ann]
    #     txt = list(chain(*txt))
    #     txt = [t for t in txt if len(t) > 0]
    #     for t in txt:
    #         for char_ in t:
    #             if char_ not in src_ocr_dict:
    #                 if char_ not in ext_ocr_list:
    #                     ext_ocr_list.append(char_)
    #
    #     # Open image
    #     # img = Image.open(image_name)
    #     img_file = synthtext_folder.joinpath(_img_file[0]).absolute().as_posix()
    #     img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    #     img_height, img_width, _ = img.shape
    #
    #     # Validation
    #     if len(np.shape(box_ann)) == 2:
    #         wordBBlen = 1
    #     else:
    #         wordBBlen = box_ann.shape[-1]
    #
    #     assert wordBBlen == len(txt)
    #     out_info = dict(
    #         file_name=_img_file[0],
    #         height=img_height,
    #         width=img_width,
    #         annotations=list()
    #     )
    #     for idx in range(wordBBlen):
    #         if wordBBlen == 1:
    #             cur_box = box_ann
    #         else:
    #             cur_box = box_ann[:, :, idx]
    #         assert cur_box.shape == (2, 4)
    #         pts1 = np.float32([[cur_box[0][0], cur_box[1][0]],
    #                            [cur_box[0][1], cur_box[1][1]],
    #                            [cur_box[0][2], cur_box[1][2]],
    #                            [cur_box[0][3], cur_box[1][3]]])
    #         instance = dict(
    #             polygon=pts1.reshape(-1).tolist(),
    #             text=txt[idx]
    #         )
    #         out_info['annotations'].append(instance)
    #     out_str = json.dumps(out_info)
    #     saver.write(out_str+'\n')
    # saver.close()
    # print(f"Total {len(img_files)} samples are converted to {out_folder}")
    # print("Saving ocr dict file...")
    # full_keys = src_ocr_dict + ext_ocr_list
    # with open(os.path.join(out_folder, 'dict.json'), 'w', encoding='utf-8') as f:
    #     json.dump(full_keys, f, ensure_ascii=False)
    # print(f"src_ocr:{src_ocr_dict}, ext_ocr:{ext_ocr_list}, full:{full_keys}")
    # print("Finish saving ocr dict file.")





if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Multi-process crop synthtext and save it to lmdb or images/text file')
    # parser.add_argument('--synthtext_folder', default=None, type=str, required=True,
    #                     help='synthtext root folder including gt.mat file, (default: None)')
    # parser.add_argument('--data_format', choices=['lmdb', 'images_with_gt_file', 'both'], default='images_with_gt_file',
    #                     type=str, required=True, help='output data format (default: images_with_gt_file)')
    # parser.add_argument('--lmdb_path', default=None, type=str,
    #                     help='output lmdb path, if data_format is lmdb, this arg must be set.  (default: None)')
    # parser.add_argument('--images_folder', default=None, type=str,
    #                     help='output cropped images root folder, '
    #                          'if data_format is not lmdb, this arg must be set. (default: None)')
    # parser.add_argument('--gt_file', default='gt.txt', type=str,
    #                     help='output gt txt file, output at images_folder/gt_file, '
    #                          'if data_format is not lmdb, this arg must be set. (default: gt.txt)')
    # args = parser.parse_args()
    # main(args)

    # # src_folder = '/apdcephfs/share_887471/common/ocr_benchmark/benchmark/SynthText'
    # # out_folder = '/apdcephfs/share_887471/common/whua/dataset/ie_e2e/SynthText'
    # src_folder = '/home/whua/code/MaskTextSpotterV3-master/datasets/synthtext/SynthText'
    # out_folder = '/home/whua/code/MaskTextSpotterV3-master/datasets/synthtext/e2e_format'
    # prepare_synthtext(src_folder, out_folder)

    # get dict
    # anno_file = '/home/whua/code/MaskTextSpotterV3-master/datasets/synthtext/e2e_format/train.txt'
    # ocr_dict = '/mnt/whua/ie_e2e_data/mm_format/table/dict.json'
    anno_file = '/apdcephfs/share_887471/common/whua/dataset/ie_e2e/SynthText/train.txt'
    # with open(ocr_dict, 'r', encoding='utf-8') as f:
    #     pre_ocr_list = json.load(f)
    # ext_ocr_list = []
    # with open(anno_file, 'r', encoding='utf-8') as f:
    #     for line_ in tqdm(f.readlines()):
    #         info_ = json.loads(line_.strip())
    #         for ann in info_['annotations']:
    #             for char_ in ann['text']:
    #                 if char_ not in pre_ocr_list:
    #                     if char_ not in ext_ocr_list:
    #                         ext_ocr_list.append(char_)
    #
    # out_dict = '/home/whua/code/MaskTextSpotterV3-master/datasets/synthtext/e2e_format/dict.json'
    # full_dict = pre_ocr_list + ext_ocr_list
    # with open(out_dict, 'w', encoding='utf-8') as f:
    #     json.dump(full_dict, f)
    # print(f"pre_ocr_dict:{pre_ocr_list}, ext:{ext_ocr_list}, full:{full_dict}")


    # max_height = 0
    # max_width = 0
    # heights = []
    # widths = []
    # with open(anno_file, 'r', encoding='utf-8') as f:
    #     for line_ in tqdm(f.readlines()):
    #         info_ = json.loads(line_.strip())
    #         heights.append(info_['height'])
    #         widths.append(info_['width'])
    #         if info_['height'] > max_height:
    #             max_height = info_['height']
    #         if info_['width'] > max_width:
    #             max_width = info_['width']
    # print(f"max_height:{max_height}, max_width:{max_width}")
    # print(f"avg_height:{np.mean(heights)}, avg_width:{np.mean(widths)}")
    invalid_instance_num = 0
    invalid_img_num = 0
    with open(anno_file, 'r', encoding='utf-8') as f:
        for line_ in tqdm(f.readlines()):
            info_ = json.loads(line_.strip())
            height = info_['height']
            width = info_['width']
            is_invalid_img = False
            for instance_ in info_['annotations']:
                poly = np.array(instance_['polygon']).reshape((-1, 2))
                is_invalid = False
                for pt in poly:
                    if pt[0] < 0:
                        is_invalid = True
                        # print(f"Invalid instance:{instance_}, height:{height}, width:{width}")
                        break
                    elif pt[1] < 0:
                        is_invalid = True
                        # print(f"Invalid instance:{instance_}, height:{height}, width:{width}")
                        break
                if is_invalid:
                    invalid_instance_num += 1
                    is_invalid_img = True
            if is_invalid_img:
                invalid_img_num += 1
    print(f"invalid_instance_num:{invalid_instance_num}, invalid_img_num:{invalid_img_num}")
    # find abnormal

