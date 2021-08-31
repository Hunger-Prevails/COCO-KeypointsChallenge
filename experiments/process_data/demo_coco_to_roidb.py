from common.pycocotools.coco import COCO
import numpy as np
import cPickle


def load_coco_annotation(annotation_path, task='det', data_dir='', need_mask=True):
    assert task in ['det', 'kps']
    coco = COCO(annotation_path)
    image_ids = coco.getImgIds()

    cats = [cat['name'] for cat in coco.loadCats(coco.getCatIds())]
    classes = ['__background__'] + cats
    num_classes = len(classes)
    class_to_ind = dict(zip(classes, xrange(num_classes)))
    class_to_coco_ind = dict(zip(cats, coco.getCatIds()))
    coco_ind_to_class_ind = dict([(class_to_coco_ind[cls], class_to_ind[cls]) for cls in classes[1:]])

    roidb = []
    for i, image_id in enumerate(image_ids):
        if i % 1000 == 0:
            print '{}/{}'.format(i, len(image_ids))
        im_ann = coco.loadImgs(image_id)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = coco.getAnnIds(imgIds=image_id, iscrowd=None)
        objs = coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if task == 'kps':
                assert obj['category_id'] == 1
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        gt_classes = np.zeros((num_objs,), dtype=np.int32)
        keypoints = np.zeros((num_objs, 51), dtype=np.float32)

        for ix, obj in enumerate(objs):
            cls = -1 if obj['iscrowd'] else coco_ind_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            if task == 'kps':
                keypoints[ix, :] = obj['keypoints']

        roi_rec = {'image': data_dir + im_ann['file_name'],
                   'height': height,
                   'width': width,
                   'boxes': boxes,
                   'gt_classes': gt_classes}
        if task == 'kps':
            roi_rec['keypoints'] = keypoints
        if need_mask:
            roi_rec['gt_masks'] = [x['segmentation'] for x in objs]

        roidb.append(roi_rec)
    return roidb


if __name__ == '__main__':
    annotation_path = '/opt/hdfs/user/xinze.chen/common/dataset/coco2017/annotations/instances_val2017.json'
    task = 'det'
    gt_roidb = load_coco_annotation(annotation_path=annotation_path, task=task)

    # annotation_path = '/opt/hdfs/user/xinze.chen/common/dataset/coco2017/annotations/person_keypoints_val2017.json'
    # task = 'kps'
    # gt_roidb = load_coco_annotation(annotation_path=annotation_path, task=task)

    save_roidb_path = './val2017_%s_gt_roidb.pkl' % task
    with open(save_roidb_path, 'wb') as fid:
        cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)