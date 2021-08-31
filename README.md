# smart_iot_multitask

smart_iot_multitask is a training framework for multitask (such as detection, keypoints and instance segmentation) based on MxNet.


Dataset and Models
------------------
  cp -r /opt/hdfs/user/xinze.chen/common/models /opt/hdfs/user/${YOUR_NAME}/common
  cp -r /opt/hdfs/user/xinze.chen/common/dataset/coco2017 /opt/hdfs/user/${YOUR_NAME}/common/dataset

  if you have your dataset, you can see experiments/process_data/demo_im2rec.py to transform your images to rec format and 
  see experiments/process_data/demo_coco_to_roidb.py to transform your labels to roidb format.

FPN
---
  cp experiments/fpn_coco/examples/config_*.py experiments/fpn_coco/config.py 
  modify config.person_name = ${YOUR_NAME}
  python experiments/fpn_coco/demo_fpn_train.py

FPN_Multitask
-------------
  cp experiments/fpn_multitask_coco/examples/config_*.py experiments/fpn_multitask_coco/config.py 
  modify config.person_name = ${YOUR_NAME}
  python python experiments/fpn_multitask_coco/demo_fpn_multitask_train.py 

keypoints
---------
  cp experiments/keypoints_coco/examples/config_*.py experiments/keypoints_coco/config.py 
  modify config.person_name = ${YOUR_NAME}
  python experiments/keypoints_coco/demo_kps_train.py 

Training on cluster
-------------------
   If you want to train on cluster, you can modify config.TRAIN.num_workers=${NUM_WORKERS} and modify function common/train_in_cluster/get_qsub_i_conf to your qsub_conf.
   Then you can run: python experiments/${TASK_NAME}/demo_train_cluster.py
