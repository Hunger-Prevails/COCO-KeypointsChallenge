import logging
from load_roidb import load_roidb
from det_eval import evaluate_recall, evaluate_ap


def load_coco_test_roidb_eval(config, annotation_path):
    # get roidb
    roidb = load_roidb(roidb_path_list=config.dataset.test_roidb_path_list,
                       imglst_path_list=config.dataset.test_imglst_path_list)
    logging.info('total num images for test: {}'.format(len(roidb)))

    from common.dataset.coco_eval import COCOEval
    imdb = COCOEval(annotation_path)

    # sample roidb
    roidb = imdb.sample_on_imdb(roidb, filter_strategy=config.TEST.filter_strategy)
    logging.info('total num images for test after sampling: {}'.format(len(roidb)))

    def eval_func(**kwargs):
        if 'rpn' in config.network.task_type and config.TEST.rpn_do_test:
            all_proposals = kwargs['all_proposals']
            for j in range(1, len(all_proposals)):
                logging.info('***************class %d****************' % j)
                gt_class_ind = j if config.network.rpn_rcnn_num_branch > 1 else None
                evaluate_recall(roidb, all_proposals[j], gt_class_ind=gt_class_ind)
        if 'rpn_rcnn' in config.network.task_type:
            imdb.evaluate_detections(kwargs['all_boxes'], alg=kwargs['alg'] + '-det')
        if 'kps' in config.network.task_type:
            imdb.evaluate_keypoints(kwargs['all_kps_results'], alg=kwargs['alg'] + '-kps')
        if 'mask' in config.network.task_type:
            imdb.evalute_sds(kwargs['all_mask_boxes'], kwargs['all_masks'], alg=kwargs['alg'] + '-segm')

    return roidb, eval_func


def load_hobot_test_roidb_eval(config):
    # get roidb
    roidb = load_roidb(roidb_path_list=config.dataset.test_roidb_path_list,
                       imglst_path_list=config.dataset.test_imglst_path_list,
                       filter_strategy=config.TEST.filter_strategy)
    logging.info('total num images for test: {}'.format(len(roidb)))

    def eval_func(**kwargs):
        if 'rpn' in config.network.task_type and config.TEST.rpn_do_test:
            all_proposals = kwargs['all_proposals']
            for j in range(1, len(all_proposals)):
                logging.info('***************class %d****************' % j)
                gt_class_ind = j if config.network.rpn_rcnn_num_branch > 1 else None
                evaluate_recall(roidb, all_proposals[j], gt_class_ind=gt_class_ind)
        if 'rpn_rcnn' in config.network.task_type:
            evaluate_ap(roidb, kwargs['all_boxes'])

    return roidb, eval_func