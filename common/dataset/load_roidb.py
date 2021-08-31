import os
import cPickle
import logging
import numpy as np


def load_roidb(roidb_path_list, imglst_path_list=None, filter_strategy=None):
    roidb_list = []
    if roidb_path_list is not None:
        for roidb_path in roidb_path_list:
            with open(roidb_path, 'rb') as fid:
                roidb = cPickle.load(fid)
            roidb_list.append(roidb)

    if filter_strategy is not None:
        roidb_list = [filter_roidb(roidb, filter_strategy) for roidb in roidb_list]

    if imglst_path_list is not None:
        add_roidb_rec_index(roidb_list, imglst_path_list)

    roidb = merge_roidb(roidb_list)
    logging.info('total num images: %d' % len(roidb))
    return roidb


def add_roidb_rec_index(roidb_list, imglst_path_list):
    assert len(roidb_list) == len(imglst_path_list)
    for i, roidb in enumerate(roidb_list):
        img_list = {}
        with open(imglst_path_list[i]) as fin:
            for line in iter(fin.readline, ''):
                line = line.strip().split('\t')
                img_list[line[-1]] = int(line[0])
        for roi_rec in roidb:
            img_name = roi_rec['image']
            if img_name not in img_list:
                img_name = os.path.basename(roi_rec['image'])
                assert img_name in img_list
            roi_rec['rec_index'] = img_list[img_name]
            roi_rec['imgrec_id'] = i


def merge_roidb(roidb_list):
    roidb = roidb_list[0]
    for r in roidb_list[1:]:
        roidb.extend(r)
    return roidb


def filter_roidb(roidb, filter_strategy, need_inds=False):
    all_choose_inds = range(len(roidb))

    def filter_roidb_func(choose_inds, filter_name, filter_func):
        if filter_name in filter_strategy and filter_strategy[filter_name]:
            num = len(choose_inds)
            choose_inds = [i for i in choose_inds if not filter_func(roidb[i])]
            num_after = len(choose_inds)
            logging.info('filter %d %s roidb entries: %d -> %d' % (num - num_after, filter_name[7:], num, num_after))
        return choose_inds

    def is_points_as_boxes(entry):
        gt_boxes = entry['boxes']
        width = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
        height = gt_boxes[:, 3] - gt_boxes[:, 1] + 1
        flag = (width > 1).all() and (height > 1).all()
        return not flag
    all_choose_inds = filter_roidb_func(all_choose_inds, 'remove_point', is_points_as_boxes)

    def is_empty_boxes(entry):
        num_valid_boxes = np.sum(entry['gt_classes'] > 0)
        return num_valid_boxes == 0    
    all_choose_inds = filter_roidb_func(all_choose_inds, 'remove_empty_boxes', is_empty_boxes)

    def is_empty_preds(entry):
        assert 'pred_boxes' in entry
        return entry['pred_boxes'].shape[0] == 0
    all_choose_inds = filter_roidb_func(all_choose_inds, 'remove_empty_preds', is_empty_preds)

    def is_single_boxes(entry):
        num_valid_boxes = np.sum(entry['gt_classes'] > 0)
        return num_valid_boxes <= 1
    all_choose_inds = filter_roidb_func(all_choose_inds, 'remove_single_boxes', is_single_boxes)

    def is_multi_boxes(entry):
        num_valid_boxes = np.sum(entry['gt_classes'] > 0)
        return num_valid_boxes > 1
    all_choose_inds = filter_roidb_func(all_choose_inds, 'remove_multi_boxes', is_multi_boxes)

    def is_empty_kps(entry):
        scores = entry['keypoints'][:, 2::3]
        scores = np.sum(scores, axis=1)
        keep = np.where(scores != 0)[0]
        entry['keypoints'] = entry['keypoints'][keep, :]
        entry['boxes'] = entry['boxes'][keep, :]
        if 'gt_classes' in entry:
            entry['gt_classes'] = entry['gt_classes'][keep]
        return np.sum(scores) == 0
    all_choose_inds = filter_roidb_func(all_choose_inds, 'remove_empty_kps', is_empty_kps)

    def is_any_unvis_kps(entry):
        scores = entry['keypoints'][:, 2::3]
        return (scores != 2).any()
    all_choose_inds = filter_roidb_func(all_choose_inds, 'remove_any_unvis_kps', is_any_unvis_kps)

    if 'max_num_images' in filter_strategy:
        max_num_images = filter_strategy['max_num_images']
        if 0 < max_num_images < len(all_choose_inds):
            num = len(all_choose_inds)
            all_choose_inds = all_choose_inds[:max_num_images]
            num_after = len(all_choose_inds)
            logging.info('filter %d roidb entries after max_num_images: %d -> %d' % (num - num_after, num, num_after))

    if 'parts' in filter_strategy:
        part_index = filter_strategy['parts'][0]
        num_parts = filter_strategy['parts'][1]
        assert part_index < num_parts
        num_inds_per_part = (len(all_choose_inds) + num_parts - 1) / num_parts
        num = len(all_choose_inds)
        all_choose_inds = all_choose_inds[part_index*num_inds_per_part: (part_index+1)*num_inds_per_part]
        num_after = len(all_choose_inds)
        logging.info('filter %d roidb entries after parts: %d -> %d' % (num - num_after, num, num_after))

    if 'indexes' in filter_strategy:
        start_index = filter_strategy['indexes'][0]
        end_index = filter_strategy['indexes'][1]
        num = len(all_choose_inds)
        assert 0 <= start_index < end_index <= num
        all_choose_inds = all_choose_inds[start_index:end_index]
        num_after = len(all_choose_inds)
        logging.info('filter %d roidb entries after indexes: %d -> %d' % (num - num_after, num, num_after))

    roidb = [roidb[i] for i in all_choose_inds]

    if need_inds:
        return roidb, all_choose_inds
    else:
        return roidb

