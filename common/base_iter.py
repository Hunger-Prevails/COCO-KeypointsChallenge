import numpy as np
import mxnet as mx
import math
import cv2
import logging
import multiprocessing
from mxnet.executor_manager import _split_input_slice
from common.processing.image import tensor_vstack


class BaseIter(mx.io.DataIter):
    def __init__(self, roidb, config, batch_size, ctx=None, work_load_list=None):
        self.roidb = roidb
        self.config = config
        self.batch_size = batch_size
        self.data = None
        self.label = None

        self.shuffle = config.TRAIN.aug_strategy.shuffle
        self.aspect_grouping = config.TRAIN.aug_strategy.aspect_grouping

        self.ctx = [mx.cpu()] if ctx is None else ctx
        if work_load_list is None:
            work_load_list = [1] * len(self.ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(self.ctx), 'Invalid settings for work load.'
        self.work_load_list = work_load_list

        self.cur = 0
        self.size = len(roidb)
        self.index = np.arange(self.size)
        self.reset()

    def load_data(self):
        self.imgrec = None
        num_imgrec = len(self.config.dataset.train_imgrec_path_list)
        if num_imgrec > 0:
            assert num_imgrec == len(self.config.dataset.train_imgidx_path_list)
            self.imgrec = []
            for i in range(num_imgrec):
                imgidx_path = self.config.dataset.train_imgidx_path_list[i]
                imgrec_path = self.config.dataset.train_imgrec_path_list[i]
                self.imgrec.append(mx.recordio.MXIndexedRecordIO(imgidx_path, imgrec_path, 'r'))
            logging.info('use imgrec for training')

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                lim = math.floor(len(horz_inds) / self.batch_size) * self.batch_size
                horz_inds = np.random.choice(horz_inds, size=int(lim), replace=False) if lim != 0 else []
                lim = math.floor(len(vert_inds) / self.batch_size) * self.batch_size
                vert_inds = np.random.choice(vert_inds, size=int(lim), replace=False) if lim != 0 else []
                inds = np.hstack((horz_inds, vert_inds))
                inds_ = np.reshape(inds, (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds.astype(np.int32)
                self.size = len(self.index)
            else:
                np.random.shuffle(self.index)

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data = self.data,
                                   label = self.label,
                                   pad = 0,
                                   provide_data = self.provide_data,
                                   provide_label = self.provide_label)
        else:
            raise StopIteration

    def get_batch(self):
        index_start = self.cur
        index_end = self.cur + self.batch_size
        roidb = [self.roidb[self.index[i]] for i in range(index_start, index_end)]
        slices = _split_input_slice(index_end - index_start, self.work_load_list)

        data_list = []
        label_list = []
        for i_slice in slices:
            i_roidb = [roidb[i] for i in range(i_slice.start, i_slice.stop)]
            for j in range(len(i_roidb)):
                data, label = self.get_one_roidb(i_roidb[j], j)
                data_list.append(data)
                label_list.append(label)

        all_data = dict()
        for name in self.data_name:
            all_data[name] = tensor_vstack([data[name] for data in data_list])

        all_label = dict()
        for name in self.label_name:
            pad = -1 if 'label' in name and 'weight' not in name else 0
            all_label[name] = tensor_vstack([label[name] for label in label_list], pad=pad)

        self.data = [mx.nd.array(all_data[name]) for name in self.data_name]
        self.label = [mx.nd.array(all_label[name]) for name in self.label_name]

    def get_one_roidb(self, roidb_j, j=0):
        return [], []


class BaseTestIter(mx.io.DataIter):
    def __init__(self, roidb, config, batch_size):
        self.roidb = roidb
        self.config = config
        self.batch_size = batch_size

        self.cur = 0
        self.size = len(self.roidb)
        self.index = np.arange(self.size)
        self.reset()

        self.load_data()

        self.data = []
        self.data_batch = []
        self.need_forward = []

        self.extra_local_vars = []

    def load_data(self):
        self.imgrec = None
        num_imgrec = len(self.config.dataset.test_imgrec_path_list)
        if num_imgrec > 0:
            assert num_imgrec == len(self.config.dataset.test_imgidx_path_list)
            self.imgrec = []
            for i in range(num_imgrec):
                imgidx_path = self.config.dataset.test_imgidx_path_list[i]
                imgrec_path = self.config.dataset.test_imgrec_path_list[i]
                self.imgrec.append(mx.recordio.MXIndexedRecordIO(imgidx_path, imgrec_path, 'r'))
            logging.info('use imgrec for test')

        self.video_reader = None
        if 'video_path' in self.config.dataset and self.config.dataset.video_path is not None:
            assert self.imgrec is None
            logging.info(self.config.dataset.video_path)
            import imageio
            self.video_reader = [imageio.get_reader(self.config.dataset.video_path)]
            logging.info('use video for test')

        self.camera_reader = None
        if 'is_camera' in self.config.dataset and self.config.dataset.is_camera:
            self.camera_reader = cv2.VideoCapture(0)
            logging.info('use camera for test')

    def reset(self):
        self.cur = 0

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return None

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data_batch, self.need_forward
        else:
            raise StopIteration

    def get_batch(self):
        self.clear_local_vars()
        index_start = self.cur
        index_end = min(index_start + self.batch_size, self.size)
        for i in range(index_start, index_end):
            if self.video_reader is not None:
                try:
                    data, need_forward = self.get_one_roidb(self.roidb[self.index[i]])
                except:
                    break
            else:
                data, need_forward = self.get_one_roidb(self.roidb[self.index[i]])
            self.data = [mx.nd.array(data[name]) for name in self.data_name]
            self.data_batch.append(mx.io.DataBatch(data=self.data, label=[], pad=0, provide_data=self.provide_data))
            self.need_forward.append(need_forward)

    def get_one_roidb(self, roidb_j, j=0):
        return []

    def clear_local_vars(self):
        self.data_batch = []
        self.need_forward = []
        self.extra_local_vars = []


def clear_shm(num_gpus, hours=2):
    import datetime
    import os
    if num_gpus == 8:
        os.system('rm /dev/shm/mx_*')
        logging.info('rm all /dev/shm/mx_* file')
        return
    time_now = datetime.datetime.now()
    shm_dir = '/dev/shm/'
    if os.path.exists(shm_dir):
        filelist = os.listdir(shm_dir)
        delete_count = 0
        for file in filelist:
            filepath = os.path.join(shm_dir, file)
            if len(file) > 3 and file[:3] == 'mx_' and os.path.isfile(filepath):
                time_delta = time_now - datetime.datetime.fromtimestamp(os.path.getctime(filepath))
                if time_delta.days > 0 or time_delta.seconds > hours * 60 * 60:
                    os.remove(filepath)
                    delete_count += 1
        logging.info('rm /dev/shm/mx_* file count: %d/%d' % (delete_count, len(filelist)))


def worker_loop(data_iter, key_queue, data_queue, shut_down):
    key_queue.cancel_join_thread()
    data_queue.cancel_join_thread()
    data_iter.load_data()
    while True:
        if shut_down.is_set():
            break
        batch_str = key_queue.get()
        if batch_str is None:
            break
        data_iter.index = [int(batch_id) for batch_id in batch_str.split()]
        assert len(data_iter.index) == data_iter.batch_size
        data_iter.cur = 0
        data_queue.put(data_iter.next())
    logging.info('goodbye')


class PrefetchingIter(mx.io.DataIter):
    def __init__(self, data_iter, num_workers=multiprocessing.cpu_count(), max_queue_size=8):
        super(PrefetchingIter, self).__init__()
        clear_shm(num_gpus=len(data_iter.ctx))
        logging.info('num workers: %d' % num_workers)

        self.data_iter = data_iter
        self.size = data_iter.size
        self.batch_size = data_iter.batch_size
        self.data_name = data_iter.data_name
        self.label_name = data_iter.label_name
        self.max_data_shape = data_iter.max_data_shape
        self.max_label_shape = data_iter.max_label_shape
        self.num_batches = self.size / self.batch_size
        assert self.size % self.batch_size == 0

        self.num_workers = num_workers
        self.workers = []
        self.cur = 0
        self.key_queue = mx.gluon.data.dataloader.Queue()
        self.data_queue = mx.gluon.data.dataloader.Queue(max_queue_size)
        self.key_queue.cancel_join_thread()
        self.data_queue.cancel_join_thread()
        self.shut_down = multiprocessing.Event()
        self._create_workers()

        import atexit
        atexit.register(lambda a: a.__del__(), self)

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        return self.data_iter.provide_label

    def _create_workers(self):
        for i in range(self.num_workers):
            worker = multiprocessing.Process(target=worker_loop,
                                             args=(self.data_iter, self.key_queue, self.data_queue, self.shut_down))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def _close_workers(self):
        for worker in self.workers:
            worker.join()
        self.workers = []

    def shutdown(self):
        self.shut_down.set()
        for i in range(len(self.workers)):
            self.key_queue.put(None)
        try:
            while not self.data_queue.empty():
                self.data_queue.get()
        except IOError:
            pass
        # self._close_workers()

    def __del__(self):
        self.shutdown()

    def reset(self):
        self.data_iter.reset()
        self.cur = 0

    def iter_next(self):
        return self.cur < self.num_batches

    def next(self):
        if self.cur == 0:
            index = self.data_iter.index.reshape((self.num_batches, self.data_iter.batch_size))
            for i in range(index.shape[0]):
                batch_str = '%d' % index[i, 0]
                for j in range(1, index.shape[1]):
                    batch_str += ' %d' % index[i, j]
                self.key_queue.put(batch_str)
        if self.iter_next():
            self.cur += 1
            return self.data_queue.get()
        else:
            raise StopIteration


if __name__ == '__main__':
    print 'cpu count: %d' % multiprocessing.cpu_count()
    class DummyIter(mx.io.DataIter):
        def __init__(self):
            self.cur = 0
            self.batch_size = 4

        def reset(self):
            self.cur = 0

        def next(self):
            index_start = self.cur
            index_end = self.cur + self.batch_size
            str = 's'
            for i in range(index_start, index_end):
                str += '_%d' % i
            return str

    dummy_iter = DummyIter()
    pref_iter = PrefetchingIter(dummy_iter, max_queue_size=6, num_workers=4)

    data_batch_dict = dict()
    for data_batch in pref_iter:
        assert data_batch not in data_batch_dict
        data_batch_dict[data_batch] = 1
        print data_batch









