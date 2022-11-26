#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/19 19:31
# @Author : WeiHua
import glob
import os
import json
import random
import shutil

import cv2
from tqdm import tqdm
import numpy as np


def get_ephoie_corpus(img_list_file, anno_dir, out_dir):
    img_list = []
    with open(img_list_file, 'r', encoding='utf-8') as f:
        for line_ in f.readlines():
            if line_.strip() == "":
                continue
            img_list.append(line_.strip())
    corpus = []
    ocr_dict = []
    for file_ in img_list:
        anno_file = os.path.join(anno_dir, f"{file_}.txt")
        with open(anno_file, 'r', encoding='utf-8') as f:
            info_ = json.load(f)
            for idx, val in info_.items():
                corpus.append(val['string'])
                for char_ in val['string']:
                    if char_ not in ocr_dict:
                        ocr_dict.append(char_)
    with open(os.path.join(out_dir, 'ephoie_corpus.txt'), 'w', encoding='utf-8') as saver:
        for str_ in corpus:
            saver.write(str_+'\n')
    ocr_dict = sorted(ocr_dict)
    with open(os.path.join(out_dir, 'ephoie_ocr_keys.txt'), 'w', encoding='utf-8') as saver:
        json.dump(ocr_dict, saver, ensure_ascii=False)
    print(f"ocr_dict:{ocr_dict}")


def convert_ephoie(list_file, img_dir, anno_dir, kv_dir, out_dir, out_file_name):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_img_dir = os.path.join(out_dir, 'image_files')
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)
    cls2idx = {'其他': 0, '年级': 1, '科目': 2, '学校': 3, '考试时间': 4, '班级': 5, '姓名': 6, '考号': 7, '分数': 8, '座号': 9, '学号': 10, '准考证号': 11}
    idx2cls = {}
    for idx, val in cls2idx.items():
        idx2cls[val] = idx
    files = []
    with open(list_file, 'r', encoding='utf-8') as f:
        for line_ in f.readlines():
            if line_.strip() == "":
                continue
            files.append(line_.strip())
    out_info = []
    for file_ in tqdm(files):
        img_path = os.path.join(img_dir, f"{file_}.jpg")
        assert os.path.exists(img_path), f"{img_path} not exists!"
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        with open(os.path.join(kv_dir, f"{file_}.txt"), 'r', encoding='utf-8') as f:
            kv_pairs = json.load(f)
        cur_info = dict(
            file_name=f"image_files/{file_}.jpg",
            height=height,
            width=width,
            entity_dict=kv_pairs,
            annotations=[]
        )
        with open(os.path.join(anno_dir, f"{file_}.txt"), 'r', encoding='utf-8') as f:
            anno_info = json.load(f)
        for idx, anno in anno_info.items():
            cur_anno = dict()
            cur_anno['polygon'] = anno['box']
            assert len(anno['box']) == 8, f"{file_} not fit!"

            # check if exists invalid text instance
            cur_box = np.array(anno['box'], dtype=np.float32)
            x, y, w, h = cv2.boundingRect(cur_box.reshape(-1, 2))
            bbox = np.array([x, y, x + w, y + h])
            bbox[0::2] = np.clip(bbox[0::2], 0, width)
            bbox[1::2] = np.clip(bbox[1::2], 0, height)
            if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
                print(f"line_:{line_}, box:{cur_box}, w:{width}, h:{width}")
                raise ValueError()

            cur_anno['text'] = anno['string']
            if anno['class'] == 'VALUE':
                appeared_cls = []
                entity_tags = []
                for tag_ in anno['tag']:
                    if tag_ == 0:
                        entity_tags.append('O')
                    else:
                        if tag_ in appeared_cls:
                            entity_tags.append(f"I-{idx2cls[tag_]}")
                        else:
                            entity_tags.append(f"B-{idx2cls[tag_]}")
                            appeared_cls.append(tag_)
            else:
                entity_tags = ['O' for _ in range(len(anno['string']))]
            cur_anno['entity'] = entity_tags
            cur_info['annotations'].append(cur_anno)
        out_info.append(cur_info)
        shutil.copyfile(img_path, os.path.join(out_img_dir, f"{file_}.jpg"))
    with open(out_file_name, 'w', encoding='utf-8') as saver:
        for info_ in out_info:
            out_str = json.dumps(info_, ensure_ascii=False)
            saver.write(out_str+'\n')





if __name__ == '__main__':
    list_file = r'E:\Dataset\KIE\EPHOIE\EPHOIE/train.txt'
    img_dir = r'E:\Dataset\KIE\EPHOIE\EPHOIE\image'
    anno_dir = r'E:\Dataset\KIE\EPHOIE\EPHOIE\label'
    kv_dir = r'E:\Dataset\KIE\EPHOIE\EPHOIE\kvpair'
    out_dir = r'E:\Dataset\KIE\EPHOIE/e2e-format'
    out_file_name = r'E:\Dataset\KIE\EPHOIE/e2e-format/train.txt'

    # list_file = r'E:\Dataset\KIE\EPHOIE\EPHOIE/test.txt'
    # img_dir = r'E:\Dataset\KIE\EPHOIE\EPHOIE\image'
    # anno_dir = r'E:\Dataset\KIE\EPHOIE\EPHOIE\label'
    # kv_dir = r'E:\Dataset\KIE\EPHOIE\EPHOIE\kvpair'
    # out_dir = r'E:\Dataset\KIE\EPHOIE/e2e-format'
    # out_file_name = r'E:\Dataset\KIE\EPHOIE/e2e-format/test.txt'

    # convert_ephoie(list_file, img_dir, anno_dir, kv_dir, out_dir, out_file_name)

    # # get ocr dict
    # avg_width = 0
    # avg_height = 0
    # img_cnt = 0
    # max_text_len = 0
    # # with open('../dict_default.json', 'r', encoding='utf-8') as f:
    # #     default_dict = json.load(f)
    # default_dict = []
    # ext_dict = []
    # files = [r'E:\Dataset\KIE\EPHOIE/e2e_format/test.txt', r'E:\Dataset\KIE\EPHOIE/e2e_format/train.txt']
    # avg_instance_num = 0
    # total_instance_num = 0
    # max_instance_num = 0
    # min_instance_num = 999
    # avg_ins_width = 0
    # avg_ins_height = 0
    # for file_ in files:
    #     with open(file_, 'r', encoding='utf-8') as f:
    #         for line in tqdm(f.readlines()):
    #             info_ = json.loads(line.strip())
    #             avg_height += info_['height']
    #             avg_width += info_['width']
    #             img_cnt += 1
    #             num_instance = len(info_['annotations'])
    #             total_instance_num += num_instance
    #             max_instance_num = max(max_instance_num, num_instance)
    #             min_instance_num = min(min_instance_num, num_instance)
    #             for anno in info_['annotations']:
    #                 avg_ins_width += (max(anno['polygon'][0::2]) - min(anno['polygon'][0::2]))
    #                 avg_ins_height += (max(anno['polygon'][1::2]) - min(anno['polygon'][1::2]))
    #                 max_text_len = max(max_text_len, len(anno['text']))
    #                 # if len(anno['text']) > 60:
    #                 #     print(anno['text'])
    #                 for char_ in anno['text']:
    #                     if char_ not in default_dict:
    #                         if char_ not in ext_dict:
    #                             ext_dict.append(char_)
    # print(f"avg:{total_instance_num/img_cnt}, max:{max_instance_num}, min:{min_instance_num}")
    # full_key = default_dict + ext_dict
    # full_key = sorted(full_key)
    # print(f"avg_height:{avg_height / img_cnt}, avg_width:{avg_width / img_cnt}")
    # print(f"ext key:{ext_dict}")
    # print(f"max_len:{max_text_len}")
    # print(f"avg_ins_height:{avg_ins_height/total_instance_num}, avg_ins_width:{avg_ins_width/total_instance_num}")
    #
    # with open(r'E:\Dataset\KIE\EPHOIE\e2e_format/dict.json', 'r', encoding='utf-8') as f:
    #     default_dict = json.load(f)
    #
    #
    # print(f"Start CHECK OCR")
    # assert len(ext_dict) == len(default_dict)
    # for char_ in ext_dict:
    #     if char_ not in default_dict:
    #         print(f"Out of Dict: {char_}")
    # print(f"FINISH")
    # import ipdb
    # ipdb.set_trace()
    # print(len(default_dict))
    # # with open('E:/Dataset/KIE/EPHOIE/e2e_format/dict.json', 'w', encoding='utf-8') as f:
    # #     json.dump(full_key, f, ensure_ascii=False)
    #
    # # cls2idx = {'其他': 0, '年级': 1, '科目': 2, '学校': 3, '考试时间': 4, '班级': 5, '姓名': 6, '考号': 7, '分数': 8, '座号': 9, '学号': 10,
    # #            '准考证号': 11}
    # # cls_list = [key_ for key_ in cls2idx.keys()]
    # # with open('E:/Dataset/KIE/EPHOIE/e2e_format/clss_list.json', 'w', encoding='utf-8') as f:
    # #     json.dump(cls_list, f, ensure_ascii=False)

    """
    avg:10.544846050870147, max:47, min:3
    avg_height:774.8781793842035, avg_width:2431.775769745649
    ext key:['座', '号', '考', '姓', '名', '班', '年', '级', '学', '校', '王', '思', '源', '荣', '成', '市', '第', '九', '中', '密', '封', '线', '地', '理', '业', '水', '平', '模', '拟', '试', '（', '共', '2', '张', '）', ':', '刘', '博', '雅', '课', '时', '跟', '踪', '检', '测', '七', '化', '反', '应', '速', '率', '9', '李', '善', '姗', '0', '1', '8', '语', '文', '下', '单', '元', '练', '习', '3', '柯', '南', '基', '础', '训', '一', '、', '字', '音', '与', '形', '6', '分', '温', '馨', '提', '示', '：', '各', '位', '同', '可', '通', '过', '"', '扫', '描', '上', '面', '的', '二', '维', '码', '获', '取', '参', '答', '案', '或', '视', '频', '讲', '解', '辅', '导', '。', '能', '好', '卷', '绩', '请', '写', '清', '看', '题', '意', '后', '再', '仔', '细', '做', '书', '工', '整', '，', '迹', '楚', '洁', '岳', '丽', '娜', '翠', '竹', '注', '事', '项', '准', '证', '；', '监', '人', '不', '读', '松', '愉', '郑', '州', '外', '国', '订', '装', '洪', '家', '达', '局', '小', '周', '+', '月', '期', '末', '全', 'J', 'I', 'N', 'P', 'A', 'O', 'F', 'U', '金', '牌', '教', '①', '和', '②', '③', '陈', '锦', '民', '新', '华', '内', '要', '连', '珍', '贵', '阳', '念', '伟', '坚', '生', '7', '场', '五', '薛', '镇', '江', '苏', '省', '常', '技', '师', '院', '在', '晰', '数', '阿', ' ', '《', '狼', '》', '青', '朱', '口', '完', '曾', '燕', '衡', '弥', '你', '行', '肖', '泽', '鹏', '马', '道', '三', '得', '林', '芳', 'S', 'W', 'E', 'D', 'Y', 'H', 'C', 'B', '精', '品', '系', '列', '.', '5', '4', '忻', '畅', '实', '验', '雪', '霞', '许', '八', '四', '黄', '福', '东', '山', '科', '册', '吴', '富', '胜', '章', '代', '相', '传', '命', '~', '节', '-', '于', '海', '次', '方', '程', '组', '乡', '杨', '志', '航', '洛', '高', '萧', '权', '梁', '抚', '顺', '韩', '智', '超', '#', '律', '综', '合', '用', '力', '觅', '严', '旺', '绍', '兴', '群', '直', '车', '蕊', '创', '类', '作', '阅', '梳', '情', '填', '空', '了', '哪', '几', '件', '谁', '干', '什', '么', '结', '果', '如', '何', '说', '有', '段', '句', '对', '表', '主', '县', '/', '区', '古', '孟', '桥', '少', '盐', '城', '部', '资', '料', '仅', '供', '交', '流', '德', '法', '治', '含', '笑', '初', '榜', '质', '量', '间', '良', '浩', '度', '自', '然', '产', '室', '编', '六', '侯', '杰', '坛', '职', '心', '彩', '十', '明', '大', '满', '其', '钟', 'R', '碧', '河', '宏', '图', '·', '步', '加', '添', '亿', '执', '信', '蔡', '晓', '兰', '铁', '路', '纪', '亭', '短', '篇', '吕', '想', '屹', '哲', '复', '旦', '附', '仑', '钧', '等', '术', '岛', '版', '素', '旭', '敏', '总', '评', '靳', '冠', '坤', '徐', '诗', '盈', '克', '拉', '玛', '依', '街', '(', ')', '长', '安', '问', '庆', '俊', '祥', '亮', '机', '关', '幼', '儿', '园', '晶', '萍', '辉', '〇', '赖', '吉', '前', '铸', '叶', '腾', '条', '粘', '贴', '处', '鸿', '洲', '门', '禁', '止', '宇', '冈', '荫', '子', '悦', '戚', '墅', '堰', '滕', '川', '远', '宜', '靖', '孝', '都', '益', '丹', '活', '页', '选', '琴', '圣', '潘', '鹤', '岗', '坊', '亚', '唐', '首', '飞', '沙', '邵', '振', '桐', '襄', '樊', '领', '潮', '先', '鸣', 'o', '蓝', '繁', '阜', '错', '误', '样', '例', '正', '确', '将', '涂', '格', '胡', '星', '点', '惠', '艺', '秀', '甘', '肃', '艳', '西', '凌', '锋', '玉', '茵', '殿', '斥', '娟', '广', '庸', 'n', 'i', 't', '英', '标', '凤', '珊', '朋', '友', '即', '束', '怎', '？', '就', '给', '我', '查', '会', '希', '望', '现', '配', '北', '云', '帆', '原', '斌', '舒', '晴', '享', '玲', '凯', '亲', '爱', '经', '半', '定', '掌', '握', '多', '知', '识', '本', '今', '天', '是', '展', '慧', '！', '很', '棒', '任', '瑶', '史', '官', '巧', '乌', '庄', '琳', '阶', '虹', '社', '凡', '养', '秋', '载', '而', '归', '歌', '舞', '冯', '豪', '积', '宁', '功', '嘉', '哈', '优', '翼', '丛', '翔', '梦', '才', '甜', '浙', '缙', '诚', '衢', '宾', '卫', '此', '只', '毅', '尧', '禾', '勿', '祝', '效', '义', '务', '育', '套', '春', '雨', '沈', '沁', '利', '旗', '龙', '游', '硕', '淮', '玄', '包', '头', '重', '冀', '邢', '台', '居', '继', '续', '努', '越', '来', '缺', '违', '己', '并', '认', '真', '核', '及', '目', '规', '置', '择', '必', '须', '使', '铅', '笔', '非', '毫', '米', '以', '黑', '色', '签', '求', '体', '按', '照', '域', '出', '无', '保', '持', '卡', '折', '叠', '破', '损', '泳', '专', '延', '百', '联', '盟', '灿', '健', '物', '世', '美', '届', '所', '最', '冲', '刺', '背', '左', '角', '两', '修', '改', '橡', '皮', '檫', '净', '他', '绘', '指', '草', '稿', '纸', '记', '液', '胶', '带', '由', '员', '右', '建', '婺', '统', '根', '据', '材', '械', '运', '动', '淑', '辛', '假', '增', '浚', '厦', '郭', '幸', 'e', '容', '轮', '丝', '日', 'w', 'z', 'b', 'x', 'c', 'm', '属', '型', '备', '梅', '永', '康', '为', '韶', '审', '耀', '钢', '“', '”', '佳', '彬', '涛', '历', '静', '制', '端', '邓', '煌', '郊', '禄', '劝', '族', '谦', '石', '湾', '麦', '乐', '奉', '象', '君', '董', '言', '潭', '征', '奔', '牛', '估', '快', '＋', '鞍', '强', '谭', '滋', '俞', '恩', '庭', '村', '谢', '雄', '普', '招', '负', '责', '桓', '邱', '迎', '辑', '崔', '孔', '宪', '毕', '升', '蕾', '否', '则', '框', '弄', '息', '骆', '•', '累', '途', '辽', '赞', '电', '磁', '感', '龚', '轩', '喜', '烟', '汤', '红', '苗', '序', '峰', '革', '茹', '廖', '铃', '屈', '已', '漳', '花', '卢', '柳', '鱼', '滩', '彪', '男', '—', '个', '雷', '气', '培', '彦', '风', '余', '森', '朵', '尔', '景', '<', '>', '范', 'Z', '昶', '齐', '公', '众', 'r', 'u', 'y', 'a', '手', '边', '晨', '冬', '价', '季', '殷', '植', '殖', '井', '芦', '邯', '郸', '矿', '傅', '亦', '潍', '唯', '波', '起', '曹', '球', '宙', '绕', '颜', '谍', '夏', '姜', '屯', '斟', '键', '扬', '仙', '孙', '占', '寻', '枣', '绝', '启', '特', '别', '每', '伯', 'Ⅰ', '擦', 'Ⅱ', '墨', '若', '未', '述', '影', '响', '杭', '尚', '贤', '环', '堂', '式', '杜', '娇', ',', '昆', '芜', '湖', '发', '威', '瑞', '兵', '莉', '戴', '莎', '军', '阮', '莹', '泉', '壮', '竞', '赛', '茶', '婉', '琼', '双', '诵', '凉', '佛', '姿', '政', '冰', '黎', '宋', '之', '苑', '桂', '欧', '壳', '薄', '勋', '疆', '昌', '宝', '这', '较', '枫', '礼', '团', '罗', '庚', '株', '施', '卓', '莆', '田', '丁', '旋', '盛', '陵', '荆', '老', '栏', '避', '免', '渊', '逢', '晋', '昭', '姚', '承', '餐', '纬', '网', '庞', '颖', '洋', '野', '开', '娥', '忠', '遥', '从', '圈', '微', '需', '营', '莲', '疏', '里', '詹', '翁', '珠', '勇', '诸', '暨', '聚', 'j', '彭', '固', '乾', '们', '接', '挑', '战', '油', '吧', '荔', '露', '埠', '聪', '汪', '济', '横', '房', '欣', '词', '京', '光', '回', '郡', '客', '观', '辰', '芹', '晚', '崖', '悄', '详', '隆', '态', '易', '杉', '转', '沛', '霖', '画', '迪', '嵊', '钱', '俐', '伶', '诺', '份', '录', '沉', '着', '收', '突', '翰', '香', '进', '铭', '柠', 'd', '钦', '芬', '万', '缘', '充', '朝', 's', '柏', '放', '熊', '白', '尖', '鉴', '除', '杂', '妮', '贞', '沿', '妙', '集', '司', '络', '随', '*', '=', '侨', '曼', '惑', '唇', '颤', '状', '惯', '终', '身', '财', '蓉', '伍', '乔', '褚', '衍', '莱', '灼', '滨', '淀', '鄞', '立', '椒', '性', '近', 'T', '鹰', '棋', '虎', '皋', '呈', '溢', '探', '究', '奥', '秘', '懂', '具', '鼎', '乃', '但', '太', '粗', '守', '涵', '祖', '润', '苹', '艾', '拼', '论', '溪', '蕴', '顾', '疑', '释', '难', '剑', "'", 'k', '宣', '鸭', '际', '毛', '境', '察', '觉', '蝶', '那', '让', '闯', '陆', '炳', '危', '独', '魏', '木', '郝', '奋', '限', '武', '骏', '驰', '[', ']', '临', '值', '臻', '妹', '荧', '绵', '湛', '蚌', '哦', '豆', '彤', '葛', '映', '仇', '逸', '策', '津', '谷', '敬', '鲍', '紫', '媚', '汉', '徒', '溶', '乘', '坐', '去', '神', '旅', '到', '磊', '汇', '存', '廊', '藏', '尾', '姑', '厉', '还', '听', '争', '异', '咏', '付', '秦', '皇', '烈', '纯', '切', '虚', '留', '痕', '刚', '赵', '巩', '简', '暑', '轻', '聂', '菲', '翟', '助', '拾', '壹', '帅', '蒋', '俏', '卉', '佩', '竟', '输', '虞', '趣', '走', '致', '巫', '宗', '裕', '丰', '寓', '袁', '渡', '慎', '朴', '捷', '傲', 'G', '吹', '蹋', '谐', '辩', '诊', '断', '把', '商', '桃', '醉', '阴', '蒙', '蒲', '泊', '变', '喻', '棠', '茂', '绎', '欢', '向', '针', '符', '泰', '该', '港', '调', '湘', '赫', '澎', '汝', '滑', '剖', '甄', '称', '母', '奶', '芝', '沧', '佰', '函', '种', '贺', '谊', '造', '灵', '霜', '府', '钻', '询', '登', 'h', 'p', '户', '始', '均', '见', '判', '播', '呵', '护', '银', '够', '补', '些', '失', '也', '差', '因', '没', '比', '便', '像', '肥', '丙', '寅', '树', '蔓', '恒', '摸', '底', '钉', '坦', '设', '淋', '著', '姬', '栗', '葆', '慈', '斯', '研', '伏', '阻', '【', '灯', '泡', '曲', '】', '岩', '韦', '印', 'g', '荷', '呼', '尹', '屏', '怀', '令', '器', '压', '菊', '深', '攀', '槐', '声', '藩', '牙', '俭', '约', '士', '界', '焦', '舰', '尤', '析', '倪', '卜', '贯', '圆', 'X', '邹', '围', '默', '揭', '诫', '°']
    max_len:76
    avg_ins_height:64.50875968008125, avg_ins_width:203.7278151580551
    """
    # get height-width distribution
    files = [r'E:\Dataset\KIE\EPHOIE/e2e_format/test.txt', r'E:\Dataset\KIE\EPHOIE/e2e_format/train.txt']
    # start from 0.01
    hw_distribution = dict()
    total_text_len = 0
    total_text_num = 0
    for file_ in files:
        with open(file_, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                info_ = json.loads(line.strip())
                for anno in info_['annotations']:
                    total_text_len += len(anno['text'])
                    total_text_num += 1
                    ins_width = (max(anno['polygon'][0::2]) - min(anno['polygon'][0::2]))
                    ins_height = (max(anno['polygon'][1::2]) - min(anno['polygon'][1::2]))
                    h_over_w = ins_height / ins_width
                    flag = int(h_over_w * 100)
                    if flag not in hw_distribution:
                        hw_distribution[flag] = 1
                    else:
                        hw_distribution[flag] += 1
    # below for pre-train only
    to_sort_hw = [[key/100, val] for key, val in hw_distribution.items()]
    to_sort_hw = sorted(to_sort_hw, key=lambda x: x[0])
    print(to_sort_hw)
    print(f"mean text len: {total_text_len/total_text_num}")
    keys = [key for key, _ in hw_distribution.items()]
    vals = [val for _, val in hw_distribution.items()]
    import matplotlib.pyplot as plt
    plt.scatter(keys, vals)
    plt.show()


    # # get EPHOIE corpus
    # img_list_file = r'E:\Dataset\KIE\EPHOIE\EPHOIE/test.txt'
    # anno_dir = r'E:\Dataset\KIE\EPHOIE\EPHOIE\label'
    # out_dir = 'E:/Dataset/KIE/EPHOIE/processed'
    # get_ephoie_corpus(img_list_file, anno_dir, out_dir)

    # # balance characters
    # with open(r"E:\Dataset\KIE\EPHOIE\processed/ephoie_ocr_keys.txt", 'r', encoding='utf-8') as f:
    #     ocr_list = json.load(f)
    # dict_ocr = {}
    # for token_ in ocr_list:
    #     dict_ocr[token_] = 0
    # ephoie_corpus = []
    # with open(r"E:\Dataset\KIE\EPHOIE\processed/ephoie_corpus.txt", 'r', encoding='utf-8') as f:
    #     for line_ in f.readlines():
    #         if line_.strip() == "":
    #             continue
    #         ephoie_corpus.append(line_.strip())
    # max_text_len = 0
    # for corpus_ in ephoie_corpus:
    #     max_text_len = max(max_text_len, len(corpus_))
    #     for char_ in corpus_:
    #         dict_ocr[char_] += 1
    # print(f"dict_ocr_cnt_pre:{dict_ocr}")
    # print(f"max_text_len:{max_text_len}")
    # # randomly generate text to balance the character count
    # max_cnt = max([val for key, val in dict_ocr.items()])
    # appendix_chars = []
    # for key_, val_ in dict_ocr.items():
    #     appendix_chars += [key_] * (max_cnt - val_)
    # appendix_corpus = []
    # while len(appendix_chars) > 0:
    #     print(len(appendix_chars))
    #     len_choice = random.randint(3, max_text_len)
    #     cur_corpus = ""
    #     for _ in range(len_choice):
    #         if len(appendix_chars) == 0:
    #             char_choice = ocr_list[random.randint(0, len(ocr_list)-1)]
    #         else:
    #             tmp = random.randint(0, len(appendix_chars)-1)
    #             char_choice = appendix_chars[tmp]
    #             appendix_chars = appendix_chars[:tmp] + appendix_chars[tmp+1:]
    #         cur_corpus += char_choice
    #     appendix_corpus.append(cur_corpus)
    # full_corpus = ephoie_corpus + appendix_corpus
    # dict_ocr_after = dict()
    # for token_ in ocr_list:
    #     dict_ocr_after[token_] = 0
    # for corpus_ in full_corpus:
    #     max_text_len = max(max_text_len, len(corpus_))
    #     for char_ in corpus_:
    #         dict_ocr_after[char_] += 1
    # print(f"dict_ocr_cnt_aft:{dict_ocr_after}")
    # print(f"max_text_len:{max_text_len}")
    # with open(r"E:\Dataset\KIE\EPHOIE\processed/chn_full_corpus.txt", 'w', encoding='utf-8') as saver:
    #     for line_ in full_corpus:
    #         saver.write(line_+'\n')


    # # restrict sequence length
    # file_ = r'E:\Dataset\KIE\Rec_Pretrain\custom_english/annotation.txt'
    # out_file = r'E:\Dataset\KIE\Rec_Pretrain\custom_english/annotation_max_145.txt'
    # out_dict_file = r'E:\Dataset\KIE\Rec_Pretrain\custom_english/dict.json'
    # avg_width = 0
    # avg_height = 0
    # img_cnt = 0
    # max_seq_len = 0
    # out_info = []
    # len_threshold = 145
    # with open('../dict_default.json', 'r', encoding='utf-8') as f:
    #     default_dict = json.load(f)
    # ext_dict = []
    # with open(file_, 'r', encoding='utf-8') as f:
    #     for line_ in tqdm(f.readlines()):
    #         if line_.strip() == "":
    #             continue
    #         info_ = json.loads(line_.strip())
    #         avg_height += info_['height']
    #         avg_width += info_['width']
    #         img_cnt += 1
    #         is_abandon = False
    #         for ins_ in info_['annotations']:
    #             if len(ins_['text']) > len_threshold:
    #                 print(f"text:{ins_['text']}")
    #                 is_abandon = True
    #                 break
    #             max_seq_len = max(max_seq_len, len(ins_['text']))
    #             for char_ in ins_['text']:
    #                 if char_ not in default_dict:
    #                     if char_ not in ext_dict:
    #                         ext_dict.append(char_)
    #             # if len(ins_['text']) > 150:
    #             #     print(ins_['text'], info_['file_name'])
    #         if not is_abandon:
    #             out_info.append(info_)
    #         else:
    #             print(info_['file_name'])
    # # with open(out_file, 'w', encoding='utf-8') as saver:
    # #     for info_ in tqdm(out_info):
    # #         out_str = json.dumps(info_)
    # #         saver.write(out_str + '\n')
    # print(f"avg_h:{avg_height/img_cnt}, avg_w:{avg_width/img_cnt}")
    # print(f"max_seq_len:{max_seq_len}")
    # whole_dict = default_dict + ext_dict
    # print(f"ext_dict:{ext_dict}")
    # # with open(out_dict_file, 'w', encoding='utf-8') as f:
    # #     json.dump(whole_dict, f, ensure_ascii=False)

