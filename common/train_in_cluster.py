import os
import time

def get_qsub_i_conf(root_dir):
    qsub_i_conf_path = root_dir + '/qsub_i.conf'
    with open(qsub_i_conf_path, 'w') as fn:
        fn.write('QUEUE=IoT\n')
        fn.write('SERVER=trainvm003.hogpu.cc\n')
    return qsub_i_conf_path

def get_now_time():
    return time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))

def get_job_shell_single_worker(root_dir, job_list):
    job_shell_path = root_dir + '/job_single_worker.sh'
    with open(job_shell_path, 'w') as fn:
        fn.write('#!/bin/bash\n')
        fn.write('cd ${TMPDIR}\n')
        fn.write('ls ./\n')
        fn.write('uname -a\n')
        fn.write('date\n')
        fn.write('env\n')
        fn.write('date\n')
        fn.write('CWD=`pwd`\n')
        fn.write('echo ${CWD}\n')
        fn.write('export PYTHONPATH=${CWD}:$PYTHONPATH\n')
        fn.write('export CLASSPATH=$HADOOP_PREFIX/lib/classpath_hdfs.jar\n')
        fn.write('export JAVA_TOOL_OPTIONS="-Xms2000m -Xmx10000m"\n')
        for job in job_list:
            fn.write('CMD="%s"\n' % job)
            fn.write('echo Running ${CMD}\n')
            fn.write('${CMD}\n')
    return job_shell_path

def get_job_shell_multi_worker(root_dir, job_list):
    for i, job in enumerate(job_list):
        exec_shell_path = root_dir + '/remote_exec_%d.sh' % i
        with open(exec_shell_path, 'w') as fn:
            fn.write('#!/bin/bash\n')
            fn.write('CWD=`pwd`\n')
            fn.write('export PYTHONPATH=${CWD}:$PYTHONPATH\n')
            fn.write('export CLASSPATH=$HADOOP_PREFIX/lib/classpath_hdfs.jar\n')
            fn.write('export JAVA_TOOL_OPTIONS="-Xms2000m -Xmx10000m"\n')
            fn.write('CMD="%s"\n' % job)
            fn.write('echo Running ${CMD}\n')
            fn.write('${CMD}\n')
        os.system('chmod -R 777 %s' % exec_shell_path)
    job_shell_path = root_dir + '/job_multi_worker.sh'
    with open(job_shell_path, 'w') as fn:
        fn.write('#!/bin/bash\n')
        fn.write('cd ${TMPDIR}\n')
        fn.write('ls ./\n')
        fn.write('uname -a\n')
        fn.write('date\n')
        fn.write('env\n')
        fn.write('date\n')
        fn.write('CWD=`pwd`\n')
        fn.write('export PYTHONPATH=${CWD}:$PYTHONPATH\n')
        for i in range(len(job_list)):
            fn.write('python  ps-lite/tracker/dmlc_mpi.py -n $((1 * $PBS_NP)) -s $PBS_NP ./remote_exec_%d.sh\n' % i)
    return job_shell_path

def qsub_i(config):
    num_gpus = len(config.TRAIN.gpus.split(','))
    root_dir = os.path.abspath(__file__).split('common')[0]
    script_home_dir = os.path.abspath(__file__).split(config.person_name)[0] + config.person_name + '/script'
    if not os.path.isdir(script_home_dir):
        os.makedirs(script_home_dir)
    qsub_i_path = script_home_dir + '/submit.sh'
    job_name = '%s_%s' % (config.exp, get_now_time())
    _conf = ' --conf %s' % get_qsub_i_conf(root_dir)
    _job_name = ' -N %s' % job_name
    _hdfs = ' --hdfs hdfs://hobot-mosdata/'
    _pods = ' --pods %d' % (num_gpus * config.TRAIN.num_workers)
    _ugi = ' --ugi `whoami`,regular-engineer'
    _hout = ' --hout /open_mlp/run_data/output/%s' % job_name
    _files = ' --files %s' % root_dir
    _walltime = ' -l walltime=%d:00:00' % (24 * 5)
    if config.TRAIN.num_workers == 1:
        _job_path = get_job_shell_single_worker(root_dir, config.job_list)
    else:
        _job_path = get_job_shell_multi_worker(root_dir, config.job_list)
    qsub_i_command = 'qsub_i' + _conf + _job_name + _hdfs + _ugi + _hout + _files + _pods + _walltime + ' ' + _job_path
    with open(qsub_i_path, 'w') as fn:
        fn.write('#!/bin/bash\n')
        fn.write('%s\n' % qsub_i_command)
    job_command = 'cd %s; bash submit.sh' % os.path.dirname(qsub_i_path)
    os.system(job_command)