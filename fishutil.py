'''Zebrafish class, also cotains some useful auxiliary functions'''

"""
Output workflow
1. instances - predicted outputs from model (outputs['instances'])
2. zebrafish instances - extract useful infos from instance (split_outputs(instances))
3. infos - [mask, [b, b, d, x], score] of each category (zebrafish_info(zebrafish_instances)), get the original data
4. Zebrafish object - each object could offer the endpoints information (zebrafish = Zebrafish(infos))
5. template - prepare the pandas template where contains the endpoints information (template = update_template(zebrafish))
6. data frame - create and input the template information line by line, the final output (df = create_pd(template))
7. Use the output to analysis
"""

import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mask_area(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    area = cv2.contourArea(contour)
    return area


def split_outputs(instances):
    zebrafish = {}
    p_bbd = instances.get('pred_boxes')
    p_bbd = [bbd.numpy().tolist() for bbd in p_bbd]
    p_masks = instances.get('pred_masks')
    p_classes = instances.get('pred_classes')
    p_classes = p_classes.numpy().tolist()
    p_scores = instances.get('scores')
    p_scores = p_scores.numpy().tolist()
    array_masks = p_masks.numpy()
    array_masks = array_masks + 0
    mask = np.uint8(array_masks)
    for i, id in enumerate(p_classes):
        info = {}
        info['category'] = p_classes[i]
        info['bbox'] = p_bbd[i]
        info['score'] = round(p_scores[i], 3)
        info['mask'] = mask[i]
        zebrafish[category_map(id)] = info
    return zebrafish


## input category id, return its name (str)
def category_map(category_id):
    category_id += 1
    categories = {'1': 'eye', '2': 'yolk', '3': 'heart', '4': 'head', '5': 'bent spine', '6': 'jaw malformation',
                  '7': 'tail',
                  '8': 'swim bladder absence', '9': 'lower jaw', '10': 'spine', '11': 'swim bladder',
                  '12': 'yolk edema',
                  '13': 'pericardial edema', '14': 'dead', '15': 'head hemorrhage', '16': 'unhatched embryo'}
    return categories[str(category_id)]


def zebrafish_info(zebrafish_instances):
    info = {'eye': [], 'yolk': [], 'heart': [], 'head': [], 'bent spine': [], 'jaw malformation': [],
            'tail': [], 'swim bladder absence': [], 'lower jaw': [], 'spine': [], 'swim bladder': [],
            'yolk edema': [], 'pericardial edema': [], 'dead': [], 'head hemorrhage': [], 'unhatched embryo': []}
    for category in zebrafish_instances:
        mask = zebrafish_instances[category].get('mask')
        bbdx = zebrafish_instances[category].get('bbox')
        score = zebrafish_instances[category].get('score')
        area = mask_area(mask)

        # compare the pred scores, if the score of the other existed category, replace the pos
        if info[category] == []:
            info[category] = [area, bbdx, score]
        else:
            if info[category][2] < score:
                info[category] = [area, bbdx, score]
            else:
                pass
    return info


def get_center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def get_left(box):
    return [box[0], (box[1] + box[3]) / 2]


def get_right(box):
    return [box[2], (box[1] + box[3]) / 2]


def get_body(info):
    body_length = 0
    if info['tail'] == [] and info['spine'] == []:
        body_length = max(
            get_distance(
                get_center(info['eye'][1]), get_right(info['bent spine'][1])
            ),
            get_distance(
                get_center(info['eye'][1]),
                get_left(info['bent spine'][1])
            )
        )
    elif info['tail'] == [] and info['spine'] != []:
        body_length = max(
            get_distance(
                get_center(info['eye'][1]),
                get_right(info['spine'][1])
            ),
            get_distance(
                get_center(info['eye'][1]),
                get_left(info['spine'][1])
            )
        )
    elif info['tail'] != []:
        body_length = get_distance(get_center(info['eye'][1]), get_center(info['tail'][1]))
    else:
        pass

    return round(body_length, 3)


def get_spine_length(info):
    spine = 0
    if info['spine'] != []:
        spine = get_distance(
            get_left(info['spine'][1]), get_right(info['spine'][1])
        )
    elif info['bent spine'] != []:
        spine = get_distance(
            get_left(info['bent spine'][1]), get_right(info['bent spine'][1])
        )
    else:
        pass

    return round(spine, 3)


def get_tail_length(info):
    tail = 0
    if info['tail'] != []:
        tail = get_distance(
            get_left(info['tail'][1]), get_right(info['tail'][1])
        )
    else:
        pass

    return round(tail, 3)


def get_curve(info):
    angle = 0
    if info['eye'] != [] and info['spine'] != []:
        if info['tail'] != []:
            angle = get_angle(get_center(info['eye'][1]), get_center(info['spine'][1]), get_center(info['tail'][1]))
        else:
            angle = max(get_angle(get_center(info['eye'][1]), get_left(info['spine'][1]), get_right(info['spine'][1])),
                        get_angle(get_center(info['eye'][1]), get_right(info['spine'][1]), get_left(info['spine'][1])))
    elif info['eye'] != [] and info['bent spine'] != []:
        if info['tail'] != []:
            angle = get_angle(get_center(info['eye'][1]), get_center(info['bent spine'][1]),
                              get_center(info['tail'][1]))
        else:
            angle = max(get_angle(get_center(info['eye'][1]), get_left(info['bent spine'][1]),
                                  get_right(info['bent spine'][1])),
                        get_angle(get_center(info['eye'][1]), get_right(info['bent spine'][1]),
                                  get_left(info['bent spine'][1])))
    return angle


def get_distance(x1, x2):
    distance = math.pow((x1[0] - x2[0]), 2) + math.pow((x1[1] - x2[1]), 2)
    distance = math.sqrt(distance)
    return round(distance, 3)


def get_angle(point_1, point_2, point_3):
    a = math.sqrt(
        (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1]))
    b = math.sqrt(
        (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1]))
    c = math.sqrt(
        (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1]))

    angle = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    return round(angle, 3)

def update_template(zebrafish_cal, scale = 0.004):
    template = {
        'eye size':0, 'head size':0, 'jaw size':0, 'heart size':0, 'yolk size':0, 'bladder size':0, 'tail length':0,
        'spine length':0, 'body length':0, 'body curvature':0, 'head hemorrhage':False, 'jaw malformation':False,
        'swim bladder absence':False, 'pericardial edema':False, 'yolk edema':False,
        'bent spine':False, 'dead':False, 'unhatched':False
    }
    template['eye size'] = float('%.3g' % (zebrafish_cal.geteye() * math.pow(scale, 2)))
    template['head size'] = float('%.3g' % (zebrafish_cal.gethead() * math.pow(scale, 2)))
    template['jaw size'] = float('%.3g' % (zebrafish_cal.getjaw() * math.pow(scale, 2)))
    template['heart size'] = float('%.3g' % (zebrafish_cal.getheart() * math.pow(scale, 2)))
    template['yolk size'] = float('%.3g' % (zebrafish_cal.getyolk() * math.pow(scale, 2)))
    template['bladder size'] = float('%.3g' % (zebrafish_cal.getbladder() * math.pow(scale, 2)))
    template['tail length'] = float('%.3g' % (zebrafish_cal.getail() * scale))
    template['spine length'] = float('%.3g' % (zebrafish_cal.getspine() * scale))
    template['body length'] = float('%.3g' % (zebrafish_cal.getlength() * scale))
    template['body curvature'] = zebrafish_cal.getcurve()
    template['head hemorrhage'] = zebrafish_cal.isheadamage()
    template['jaw malformation'] = zebrafish_cal.isjawmal()
    template['swim bladder absence'] = zebrafish_cal.isabsence()
    template['pericardial edema'] = zebrafish_cal.ispedema()
    template['yolk edema'] = zebrafish_cal.isyedema()
    template['bent spine'] = zebrafish_cal.isbent()
    template['dead'] = zebrafish_cal.isdead()
    template['unhatched'] = zebrafish_cal.isembryo()

    return template


def create_pd():
    df = pd.DataFrame(columns=[
        'eye size', 'head size', 'jaw size', 'heart size', 'yolk size', 'bladder size', 'tail length',
        'spine length', 'body length', 'body curvature', 'head hemorrhage', 'jaw malformation',
        'swim bladder absence', 'pericardial edema', 'yolk edema', 'bent spine', 'dead', 'unhatched'
    ])

    return df

def plot(df):
    plt.figure(figsize=(16, 9), dpi=80)

    plt.subplot(221)
    plt.scatter(df['body length'], df['eye size'], color='g', alpha=0.5, label='eye')
    plt.xlabel('body length / mm', fontsize=10)
    plt.ylabel('eye size / mm2', fontsize=10)
    plt.legend()
    plt.subplot(222)
    plt.scatter(df['body length'], df['head size'], color='y', alpha=0.5, label='head')
    plt.xlabel('body length / mm', fontsize=10)
    plt.ylabel('head size / mm2', fontsize=10)
    plt.legend()
    plt.subplot(223)
    plt.scatter(df['body length'], df['heart size'], color='r', alpha=0.5, label='heart')
    plt.xlabel('body length / mm', fontsize=10)
    plt.ylabel('heart size / mm2', fontsize=10)
    plt.legend()
    plt.subplot(224)
    plt.scatter(df['body length'], df['yolk size'], color='b', alpha=0.5, label='yolk')
    plt.xlabel('body length / mm', fontsize=10)
    plt.ylabel('yolk size / mm2', fontsize=10)
    plt.legend()

    plt.tight_layout()
    plt.savefig('normal.png', transparent=False, dpi=80, bbox_inches='tight')
    plt.show()