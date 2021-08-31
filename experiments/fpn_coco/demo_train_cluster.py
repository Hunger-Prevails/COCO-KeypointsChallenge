import init_paths
from config import config
from common.train_in_cluster import qsub_i

if __name__ == '__main__':
    qsub_i(config)