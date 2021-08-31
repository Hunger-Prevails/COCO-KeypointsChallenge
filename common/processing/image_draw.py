from random import random as rand
import cv2
import numpy as np
import math

def draw_points(im, points, point_color):
    if point_color is None:
        color = (rand() * 255, rand() * 255, rand() * 255)
    else:
        color = point_color
    for i in range(points.shape[0]):
        cv2.circle(im, (points[i, 0], points[i, 1]), 1, color=color, thickness=1)
    return im

def draw_box(im, box, box_color=None):
    # box: (4,)
    color = (rand() * 255, rand() * 255, rand() * 255) if box_color is None else box_color
    cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=color, thickness=2)
    return im

def draw_kps(im, kps, skeleton=None, point_color=None, skeleton_color=None, kps_thresh=0, show_num=False):
    # kps: (num_kps * 3, )
    kps = kps.reshape((-1, 3))
    for j in range(kps.shape[0]):
        x = int(kps[j, 0] + 0.5)
        y = int(kps[j, 1] + 0.5)
        v = kps[j, 2]
        if kps_thresh < v < 3:
            if point_color is None:
                color = (rand() * 255, rand() * 255, rand() * 255)
            elif isinstance(point_color, list):
                color = point_color[j]
            else:
                color = point_color
            # cv2.circle(im, (x, y), 3, color=color, thickness=2)
            cv2.circle(im, (x, y), 2, color=color, thickness=2)
            if show_num:
                # cv2.putText(im, '%.2f' % v, (x+3, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color, 1)
                cv2.putText(im, '%d' % j, (x+3, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color, 1)
    if skeleton is not None:
        for j in range(skeleton.shape[0]):
            p1 = skeleton[j, 0]
            p2 = skeleton[j, 1]
            x1 = int(kps[p1, 0] + 0.5)
            y1 = int(kps[p1, 1] + 0.5)
            x2 = int(kps[p2, 0] + 0.5)
            y2 = int(kps[p2, 1] + 0.5)
            if kps_thresh < kps[p1, 2] < 3 and kps_thresh < kps[p2, 2] < 3:
                if skeleton_color is None:
                    color = (rand() * 255, rand() * 255, rand() * 255)
                elif isinstance(skeleton_color, list):
                    color = skeleton_color[j]
                else:
                    color = skeleton_color
                cv2.line(im, (x1, y1), (x2, y2), color=color, thickness=2)

                # cx = (x1 + x2) / 2
                # cy = (y1 + y2) / 2
                # length = np.linalg.norm([x1 - x2, y1 - y2])
                # angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
                # polygon = cv2.ellipse2Poly((int(cx), int(cy)), (int(length/2), 2), int(angle), 0, 360, 1)
                # cv2.fillConvexPoly(im, polygon, color)
    return im

def get_edge_points(mask, min_dist=25):
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    if len(contours) > 0:
        num_points = 0
        points = 0
        for j in range(len(contours)):
            if len(contours[j]) > num_points:
                num_points = len(contours[j])
                points = contours[j]
        points = points.reshape((-1, 2))
    else:
        points = np.zeros((0, 2), dtype=np.int32)
    if points.shape[0] > 0:
        mask_len = mask.shape[0] + mask.shape[1]
        min_num_points = 15
        if mask_len / min_num_points < min_dist:
            min_dist = mask_len / min_num_points
        new_points = []
        last_point = [points[0, 0], points[0, 1]]
        new_points.append(last_point)
        for i in range(1, points.shape[0]):
            dist = math.sqrt((points[i, 0] - last_point[0]) ** 2 + (points[i, 1] - last_point[1]) ** 2)
            if dist >= min_dist:
                last_point = [points[i, 0], points[i, 1]]
                new_points.append(last_point)
        points = np.array(new_points)
    # print len(points)
    return points

def get_edge_mask(mask, edge_size=5):
    pad = edge_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_size, edge_size))
    new_mask = np.zeros((mask.shape[0] + 2 * pad, mask.shape[1] + 2 * pad), dtype=mask.dtype)
    new_mask[pad:mask.shape[0] + pad, pad:mask.shape[1] + pad] = mask
    edge_mask = new_mask - cv2.erode(new_mask, kernel)
    return edge_mask[pad:mask.shape[0] + pad, pad:mask.shape[1] + pad]

def get_mask(edge_points, mask_height, mask_width):
    from skimage.draw import polygon
    edge_mask = np.zeros((mask_height, mask_width), dtype=np.bool)
    rr, cc = polygon(edge_points[:, 1], edge_points[:, 0])
    edge_mask[rr, cc] = 1
    return edge_mask

def draw_mask(im, box, mask, mask_color=None, mask_edge_color=None, scale=0.25, binary_thresh=0.4):
    if mask_color is None:
        mask_color = (rand() * 255, rand() * 255, rand() * 255)
    mask_color = np.array(mask_color).reshape((1, 1, 3))
    if mask_edge_color is None:
        mask_edge_color = (200, 200, 200)
    mask_edge_color = np.array(mask_edge_color).reshape((1, 1, 3))
    if mask.shape[:2] != im.shape[:2]:
        mask = cv2.resize(mask, (box[2] - box[0] + 1, box[3] - box[1] + 1), interpolation=cv2.INTER_LINEAR)
        mask = (mask >= binary_thresh).astype(np.uint8)
        roi_im = im[box[1]:box[3] + 1, box[0]:box[2] + 1, :]
        mask_ = mask[:, :, np.newaxis]
        roi_im[:] = roi_im - scale * roi_im * mask_ + scale * mask_color * mask_

        edge_mask = get_edge_mask(mask)
        edge_mask_ = edge_mask[:, :, np.newaxis]
        roi_im[:] = roi_im - roi_im * edge_mask_ + mask_edge_color * edge_mask_

        # edge_points = get_edge_points(mask[:, :, 0])
        # mask = get_mask(edge_points, mask.shape[0], mask.shape[1])
        # roi_im[:] = draw_points(roi_im, edge_points, point_color=(0, 0, 255))
    else:
        mask = mask >= binary_thresh
        mask = mask[:, :, np.newaxis]
        im = im - scale * im * mask + scale * mask_color * mask
    return im

def draw_all(im, all_boxes=None, all_kps=None, skeleton=None, all_mask_boxes=None, all_masks=None,
             box_color=None, point_color=None, skeleton_color=None, mask_color=None,
             kps_thresh=0, show_num=False, do_draw_box=True):
    # im: (h, w, 3)
    # all_boxes: (num_boxes, 4)
    # all_kps: (num_boxes, num_kps*3)
    # all_masks: (num_boxes, h, w) or (num_boxes, mask_h, mask_w)
    num_boxes = 0
    if all_boxes is not None:
        num_boxes = len(all_boxes)
        all_boxes = np.round(all_boxes).astype(int)
    if all_kps is not None:
        num_boxes = len(all_kps)
    if all_masks is not None:
        assert len(all_mask_boxes) == len(all_masks)
        all_mask_boxes = np.round(all_mask_boxes).astype(int)
        num_boxes = len(all_masks)
    if num_boxes == 0:
        return im
    im_draw = im.copy()
    for i in range(num_boxes):
        color = (rand() * 255, rand() * 255, rand() * 255)
        if all_boxes is not None and do_draw_box:
            if box_color is None:
                box_color_i = color
            elif isinstance(box_color, list):
                box_color_i = box_color[i]
            else:
                box_color_i = box_color
            im_draw = draw_box(im_draw, all_boxes[i], box_color=box_color_i)
        if all_masks is not None:
            if mask_color is None:
                mask_color_i = color
            elif isinstance(mask_color, list):
                mask_color_i = mask_color[i]
            else:
                mask_color_i = mask_color
            im_draw = draw_mask(im_draw, all_mask_boxes[i], all_masks[i], mask_color=mask_color_i)
        if all_kps is not None:
            point_color_i = point_color if point_color is not None else color
            skeleton_color_i = skeleton_color if skeleton_color is not None else color
            im_draw = draw_kps(im_draw, all_kps[i], skeleton=skeleton,
                               point_color=point_color_i, skeleton_color=skeleton_color_i,
                               kps_thresh=kps_thresh, show_num=show_num)
    return im_draw

def draw_roidb(roidb, config):
    import mxnet as mx
    import os
    from common.utils.utils import empty_dir
    from common.processing.image_aug import get_image
    from common.processing.mask_transform import polys_or_rles_to_boxes, polys_or_rles_to_masks
    from common.processing.bbox_transform import clip_boxes
    from keypoints.kps_get_batch import kps_generate_new_rois
    from common.processing.image_aug import aug_data_func

    imgrec = []
    for i in range(len(config.dataset.train_imgrec_path)):
        imgidx_path = config.dataset.train_imgidx_path[i]
        imgrec_path = config.dataset.train_imgrec_path[i]
        imgrec.append(mx.recordio.MXIndexedRecordIO(imgidx_path, imgrec_path, 'r'))

    vis_kwargs = dict()
    vis_kwargs['point_color'] = (0, 255, 255)
    vis_kwargs['skeleton_color'] = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                                    [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                                    [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [0, 0, 255]]
    # vis_kwargs['box_color'] = (0, 255, 255)
    # vis_kwargs['mask_color'] = (240, 240, 240)

    home_dir = os.path.abspath(__file__).split('xinze.chen')[0] + 'xinze.chen'
    im_save_dir = home_dir + '/experiments/images_coco/'
    empty_dir(im_save_dir)

    mask_height = 28
    mask_width = 28

    num_roidb = len(roidb)
    for i in range(num_roidb):
        if i % 100 == 0:
            print '%d/%d' % (i, num_roidb)
        roi_rec = roidb[i]
        im = get_image(roi_rec, imgrec=imgrec)
        assert im.shape[0] == roi_rec['height']
        assert im.shape[1] == roi_rec['width']
        gt_classes = roi_rec['gt_classes']
        gt_boxes = roi_rec['boxes']
        gt_keypoints = roi_rec['keypoints']
        gt_polys_or_rles = roi_rec['gt_masks']
        num_boxes = len(gt_boxes)
        assert num_boxes == len(gt_keypoints) and num_boxes == len(gt_polys_or_rles)
        keep = np.where(gt_classes != -1)[0]
        gt_boxes = gt_boxes[keep, :]
        gt_keypoints = gt_keypoints[keep, :]
        gt_polys_or_rles = [gt_polys_or_rles[_] for _ in keep]

        res_dict = aug_data_func(img=im,
                                 all_boxes=gt_boxes,
                                 all_polys=gt_polys_or_rles,
                                 flip=True,
                                 scales=config.TRAIN.aug_strategy.scales)
        im = res_dict['img']
        gt_boxes = res_dict['all_boxes']
        gt_polys_or_rles = res_dict['all_polys']

        # new_boxes, _ = kps_generate_new_rois(gt_boxes, gt_boxes.shape[0], rescale_factor=[-0.5, 0.5], jitter_center=True)
        # new_boxes = clip_boxes(new_boxes, (im.shape[0], im.shape[1]))

        gt_boxes = clip_boxes(gt_boxes, (im.shape[0], im.shape[1]))
        gt_masks = polys_or_rles_to_masks(gt_polys_or_rles, gt_boxes, mask_height, mask_width)

        # masks = mask_coco2voc(gt_polys_or_rles, roi_rec['height'], roi_rec['width'])
        # gt_masks = np.zeros((gt_boxes.shape[0], 28, 28), np.float32)
        # for i in range(gt_boxes.shape[0]):
        #     box_i = roi_rec['boxes'][i].astype(int)
        #     masks_i = masks[i, box_i[1]:box_i[3] + 1, box_i[0]:box_i[2] + 1]
        #     gt_masks[i, :, :] = cv2.resize(masks_i.astype(np.float32), (28, 28))

        # draw ori
        im_draw = draw_all(im=im,
                           all_boxes=gt_boxes,
                           all_masks=gt_masks,
                           skeleton=config.dataset.kps_skeleton,
                           kps_thresh=0.2,
                           show_num=False,
                           **vis_kwargs)

        # ss = roi_rec['image'].split('/')
        # assert len(ss) == 4
        # im_save_path = os.path.join(im_save_dir, ss[1], ss[2], ss[3])
        # assert not os.path.exists(im_save_path)
        # if not os.path.exists(os.path.dirname(im_save_path)):
        #     os.makedirs(os.path.dirname(im_save_path))
        im_base_name = os.path.splitext(os.path.basename(roi_rec['image']))[0]
        im_save_path = im_save_dir + im_base_name + '_draw.jpg'
        cv2.imwrite(im_save_path, im_draw)





