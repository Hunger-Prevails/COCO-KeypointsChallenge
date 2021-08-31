import numpy as np
import random
import math
import cv2
import mxnet as mx
from image import resize, AugImage


def get_image(roi_rec, imgrec=None, video_reader=None, camera_reader=None):
    if imgrec is not None:
        _, img = mx.recordio.unpack_img(imgrec[roi_rec['imgrec_id']].read_idx(roi_rec['rec_index']), cv2.IMREAD_COLOR)
        # img1 = cv2.imread('/data-sda/xinze.chen/common/dataset/PoseTrack/' + roi_rec['image'], cv2.IMREAD_COLOR)
        # assert np.sum(abs(img - img1)) == 0
    elif video_reader is not None:
        img = np.array(video_reader[roi_rec['video_id']].get_data(roi_rec['video_index']))
        img = img[:, :, ::-1]  # (RGB --> BGR)
    elif camera_reader is not None:
        _, img = camera_reader.read()
    else:
        img = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR)
    return img


def flip_image(src_img):
    # src_img: (h, w, c)
    return cv2.flip(src_img, 1)


def flip_boxes(src_boxes, img_width):
    # src_boxes: (num_boxes, 4)  [x1, y1, x2, y2]
    dst_boxes = src_boxes.copy()
    dst_boxes[:, 0] = img_width - src_boxes[:, 2] - 1.0
    dst_boxes[:, 2] = img_width - src_boxes[:, 0] - 1.0
    return dst_boxes


def flip_coco_kps(src_kps, img_width):
    # src_kps: (num_boxes, num_kps*3)  [x1, y1, v1]
    num_kps = src_kps.shape[1] / 3
    assert num_kps == 17
    dst_kps_tmp = src_kps.copy()
    dst_kps_tmp[:, ::3] = img_width - src_kps[:, ::3] - 1.0
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    l_index = [LEFT_EYE, LEFT_EAR, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]
    r_index = [RIGHT_EYE, RIGHT_EAR, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]
    dst_kps = dst_kps_tmp.copy()
    for idx in range(len(l_index)):
        dst_kps[:, l_index[idx]*3: l_index[idx]*3+3] = dst_kps_tmp[:, r_index[idx]*3: r_index[idx]*3+3]
        dst_kps[:, r_index[idx]*3: r_index[idx]*3+3] = dst_kps_tmp[:, l_index[idx]*3: l_index[idx]*3+3]
    return dst_kps


def flip_polys_or_rles(polys_or_rles, img_width):
    flipped_polys_or_rles = []
    for i, ann in enumerate(polys_or_rles):
        if type(ann) == list:
            # Polygon format
            flipped_polys = []
            for poly in ann:
                flipped_poly = np.array(poly, dtype=np.float32)
                flipped_poly[0::2] = img_width - flipped_poly[0::2] - 1
                flipped_polys.append(flipped_poly.tolist())
            flipped_polys_or_rles.append(flipped_polys)
        else:
            # RLE format
            assert False
    return flipped_polys_or_rles


def rotate_image(src, angle):
    w = src.shape[1]
    h = src.shape[0]
    radian = angle / 180.0 * math.pi
    radian_sin = math.sin(radian)
    radian_cos = math.cos(radian)
    new_w = int(abs(radian_cos * w) + abs(radian_sin * h))
    new_h = int(abs(radian_sin * w) + abs(radian_cos * h))
    rot_mat = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle, 1.0)
    rot_mat[0, 2] += (new_w - w) / 2.0
    rot_mat[1, 2] += (new_h - h) / 2.0
    dst_img = cv2.warpAffine(src, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)
    return dst_img


def rotate_points(src_points, angle, src_img_shape, dst_img_shape, do_clip=True):
    # src_points: (num_points, 2)
    # img_shape: [h, w, c]
    src_img_center = [src_img_shape[1] / 2.0, src_img_shape[0] / 2.0]
    dst_img_center = [dst_img_shape[1] / 2.0, dst_img_shape[0] / 2.0]
    radian = angle / 180.0 * math.pi
    radian_sin = math.sin(radian)
    radian_cos = math.cos(radian)
    dst_points = np.zeros(src_points.shape, dtype=src_points.dtype)
    src_x = src_points[:, 0] - src_img_center[0]
    src_y = src_points[:, 1] - src_img_center[1]
    dst_points[:, 0] = radian_cos * src_x + radian_sin * src_y
    dst_points[:, 1] = -radian_sin * src_x + radian_cos * src_y
    dst_points[:, 0] += dst_img_center[0]
    dst_points[:, 1] += dst_img_center[1]
    if do_clip:
        dst_points[:, 0] = np.clip(dst_points[:, 0], 0, dst_img_shape[1] - 1)
        dst_points[:, 1] = np.clip(dst_points[:, 1], 0, dst_img_shape[0] - 1)
    return dst_points


def rotate_boxes(src_boxes, angle, src_img_shape, dst_img_shape):
    # src_boxes: (num_boxes, 4)  [x1, y1, x2, y2]
    num_boxes = src_boxes.shape[0]
    x1 = src_boxes[:, 0, np.newaxis]
    y1 = src_boxes[:, 1, np.newaxis]
    x2 = src_boxes[:, 2, np.newaxis]
    y2 = src_boxes[:, 3, np.newaxis]
    lt = np.hstack([x1, y1])
    rt = np.hstack([x2, y1])
    lb = np.hstack([x1, y2])
    rb = np.hstack([x2, y2])
    src_points = np.vstack([lt, rt, lb, rb])
    dst_points = rotate_points(src_points, angle, src_img_shape, dst_img_shape)
    dst_lt = dst_points[:num_boxes, :]
    dst_rt = dst_points[num_boxes:num_boxes*2, :]
    dst_lb = dst_points[num_boxes*2:num_boxes*3, :]
    dst_rb = dst_points[num_boxes*3:, :]
    dst_boxes = np.zeros(src_boxes.shape, dtype=src_boxes.dtype)
    dst_boxes[:, 0] = np.minimum(dst_lt[:, 0], dst_lb[:, 0])
    dst_boxes[:, 1] = np.minimum(dst_lt[:, 1], dst_rt[:, 1])
    dst_boxes[:, 2] = np.maximum(dst_rt[:, 0], dst_rb[:, 0])
    dst_boxes[:, 3] = np.maximum(dst_lb[:, 1], dst_rb[:, 1])
    return dst_boxes


def rotate_kps(src_kps, angle, src_img_shape, dst_img_shape):
    # src_kps: (num_boxes, num_kps * 3)  [x1, y1, v1]
    num_boxes = src_kps.shape[0]
    src_kps = src_kps.reshape((-1, 3))
    dst_kps = src_kps.copy()
    dst_kps[:, :2] = rotate_points(src_kps[:, :2], angle, src_img_shape, dst_img_shape)
    return dst_kps.reshape((num_boxes, -1))


def aug_data_func(img, all_boxes=None, all_kps=None, all_polys=None, all_masks=None,
                  flip=False, rotated_angle_range=0, scales=None,
                  image_stride=0, aug_img=False, use_y=False):
    res_dict = dict()
    if flip and random.randint(0, 1) == 1:
        img = flip_image(img)
        if all_boxes is not None and len(all_boxes) > 0:
            all_boxes = flip_boxes(all_boxes, img.shape[1])
        if all_kps is not None and len(all_kps) > 0:
            all_kps = flip_coco_kps(all_kps, img.shape[1])
        if all_polys is not None and len(all_polys) > 0:
            all_polys = flip_polys_or_rles(all_polys, img.shape[1])
        if all_masks is not None and len(all_masks) > 0:
            for j in range(all_masks.shape[0]):
                all_masks[j, :, :] = flip_image(all_masks[j, :, :])
        res_dict['flip'] = True
    else:
        res_dict['flip'] = False

    if rotated_angle_range > 0:
        assert all_polys is None
        rotated_angle = random.randint(-rotated_angle_range, rotated_angle_range)
        ori_img_shape = img.shape
        img = rotate_image(img, rotated_angle)
        if all_boxes is not None and len(all_boxes) > 0:
            all_boxes = rotate_boxes(all_boxes, rotated_angle, ori_img_shape, img.shape)
        if all_kps is not None and len(all_kps) > 0:
            all_kps = rotate_kps(all_kps, rotated_angle, ori_img_shape, img.shape)
        if all_masks is not None and len(all_masks) > 0:
            num_mask = all_masks.shape[0]
            new_all_masks = np.zeros((num_mask, img.shape[0], img.shape[1]))
            for j in range(num_mask):
                new_all_masks[j, :, :] = rotate_image(all_masks[j, :, :], rotated_angle)
            all_masks = new_all_masks
        res_dict['rotated_angle'] = rotated_angle
    else:
        res_dict['rotated_angle'] = 0

    if scales is not None:
        scale_ind = random.randint(0, len(scales) - 1)
        target_size = scales[scale_ind][0]
        max_size = scales[scale_ind][1]
        img, img_scale = resize(img, target_size, max_size, stride=image_stride)
        if all_boxes is not None and len(all_boxes) > 0:
            all_boxes = all_boxes * img_scale
        if all_kps is not None and len(all_kps) > 0:
            kps_len = len(all_kps)
            all_kps = all_kps.reshape((-1, 3))
            all_kps[:, :2] = all_kps[:, :2] * img_scale
            all_kps = all_kps.reshape((kps_len, -1))
        if all_polys is not None and len(all_polys) > 0:
            for i, ann in enumerate(all_polys):
                if type(ann) == list:
                    # Polygon format
                    for j, poly in enumerate(ann):
                        poly = np.array(poly, dtype=np.float32)
                        poly *= img_scale
                        all_polys[i][j] = poly.tolist()
                else:
                    # RLE format
                    assert False
        if all_masks is not None and len(all_masks) > 0:
            num_mask = all_masks.shape[0]
            new_all_masks = np.zeros((num_mask, img.shape[0], img.shape[1]))
            for j in range(num_mask):
                new_mask = cv2.resize(all_masks[j, :, :], None, None, fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LINEAR)
                new_all_masks[j, :new_mask.shape[0], :new_mask.shape[1]] = new_mask
            all_masks = new_all_masks
        res_dict['img_scale'] = img_scale
    else:
        res_dict['img_scale'] = 1.0

    if aug_img:
        img = AugImage(img)
    if use_y:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 0]

    res_dict['img'] = img
    res_dict['all_boxes'] = all_boxes
    res_dict['all_kps'] = all_kps
    res_dict['all_polys'] = all_polys
    res_dict['all_masks'] = all_masks
    return res_dict
