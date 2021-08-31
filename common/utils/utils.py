import os
import numpy as np
import mxnet as mx
import logging
import cPickle


def makediv(input, stride=0):
    if stride == 0:
        return input
    else:
        return int(np.ceil(input / float(stride)) * stride)


def get_kv(num_device):
    if num_device > 1:
        if num_device >= 4:
            kv = mx.kvstore.create('device')
        else:
            kv = mx.kvstore.create('local')
    else:
        kv = None
    return kv


def create_logger(log_path=None, log_format='%(asctime)-15s %(message)s'):
    if log_path is not None:
        if os.path.exists(log_path):
            os.remove(log_path)
        log_dir = os.path.dirname(log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger = logging.getLogger()
        logger.handlers = []
        formatter = logging.Formatter(log_format)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, format=log_format)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, format=log_format)
    return logger


def make_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


# def empty_dir(dir):
#     if os.path.exists(dir):
#         filelist = os.listdir(dir)
#         for file in filelist:
#             filepath = os.path.join(dir, file)
#             if os.path.isfile(filepath):
#                 os.remove(filepath)
#     else:
#         os.makedirs(dir)

def empty_dir(dir):
    if os.path.exists(dir):
        os.system('rm -r %s' % dir)
    os.system('mkdir %s' % dir)


def copy_file(src_file, dst_file):
    # copy_command = 'hdfs dfs -put %s %s.hdfs' % (src_file, dst_file)
    # logging.info(copy_command)
    # os.system(copy_command)
    # copy_command = 'hadoop fs -put %s %s.hadoop' % (src_file, dst_file)
    # logging.info(copy_command)
    # os.system(copy_command)
    copy_command = 'cp -rf %s %s' % (src_file, dst_file)
    logging.info(copy_command)
    os.system(copy_command)
    os.system(copy_command)
    chmod_command = 'chmod -R 777 %s' % dst_file
    os.system(chmod_command)


def add_music_to_video(src_video_path, dst_video_path):
    aud_path = src_video_path[:-4] + '.m4a'
    save_video_music_path = dst_video_path[:-4] + '_music' + dst_video_path[-4:]
    if os.path.exists(save_video_music_path):
        cmd = 'rm -f %s' % save_video_music_path
        os.system(cmd)
    cmd = "ffmpeg -i %s -vn -y -acodec copy %s" % (src_video_path, aud_path)
    os.system(cmd)
    cmd = "ffmpeg -i %s -i %s -vcodec copy -acodec copy %s" % (dst_video_path, aud_path, save_video_music_path)
    os.system(cmd)
    cmd = 'rm -f %s' % aud_path
    os.system(cmd)


def load_param(prefix, epoch):
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def serialize(obj):
    """Serialize a Python object using pickle and encode it as an array of
    float32 values so that it can be feed into the workspace. See deserialize().
    """
    return np.fromstring(cPickle.dumps(obj), dtype=np.uint8).astype(np.float32)


def deserialize(arr):
    """Unserialize a Python object from an array of float32 values fetched from
    a workspace. See serialize().
    """
    return cPickle.loads(arr.astype(np.uint8).tobytes())
