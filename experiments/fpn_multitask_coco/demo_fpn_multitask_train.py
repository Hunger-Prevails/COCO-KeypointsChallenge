import init_paths
import copy
from config import config as cfg
from fpn.fpn_iter import FPNIter
from common.train_net import train_net
from common.dataset.load_roidb_eval import load_coco_test_roidb_eval
from fpn_multitask.fpn_multitask_predict import eval_during_training_func


def main(config):
    epoch_end_callback = []
    if config.TRAIN.do_eval_during_training:
        test_roidb, eval_func = load_coco_test_roidb_eval(config, annotation_path=config.dataset.test_coco_annotation_path)
        epoch_end_callback = eval_during_training_func(roidb=test_roidb, eval_func=eval_func, config=config)
    train_net(config=config,
              DataTrainIter=FPNIter,
              epoch_end_callback=epoch_end_callback)


if __name__ == '__main__':
    config = cfg
    main(config)

