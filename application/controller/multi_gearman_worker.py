# -*- coding: utf-8 -*-

import os
import signal
import threading
import multiprocessing

import redis
from gearman.worker import GearmanWorker, POLL_TIMEOUT_IN_SECONDS

WORKER_PROCESS_PID = '/tmp/multi_gearman_worker.pid'


class MultiGearmanWorker(GearmanWorker):
    """ 多进程gearman worker"""
    def __init__(self, host_list=None, redis_host=None, redis_port=None, pid=WORKER_PROCESS_PID):
        super(MultiGearmanWorker, self).__init__(host_list=host_list)
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.pid = pid

    def work(self, poll_timeout=POLL_TIMEOUT_IN_SECONDS, process=multiprocessing.cpu_count()):
        """
        开始作业,进程阻塞
        :param poll_timeout: int gearman的连接时间,时间越短子进程worker召回越快但请求越频繁
        :param process: int 工作进程数,默认为CPU个数
        :return:
        """
        print('Clear last process.')
        self.gearman_worker_exit()
        print('Ready to start %d process for work.' % process)
        gm_poll = multiprocessing.Pool(process)
        for x in range(0, process):
            gm_poll.apply_async(gearman_work, (self, poll_timeout, self.pid))
        gm_poll.close()
        gm_poll.join()
        # 正常退出则删除子进程PID文件
        if os.path.isfile(self.pid):
            os.remove(self.pid)

        print('Multi gearman worker exit.')

    def gearman_worker_exit(self):
        """ 结束子进程 """
        if not os.path.isfile(self.pid):
            return True

        with open(self.pid, 'r+') as f:
            for pid in f.readlines():
                pid = int(pid)
                try:
                    os.kill(pid, signal.SIGKILL)
                    print('Kill process %d.' % pid)
                except OSError:
                    print('Process %d not exists' % pid)
                    continue
        os.remove(self.pid)
        print('Remove process pid file.')
        return True

# 子进程使用的gearman工作开关标识
GEARMAN_CONTINUE_WORK = True


def gearman_work(gm_worker, poll_timeout=POLL_TIMEOUT_IN_SECONDS, pid=WORKER_PROCESS_PID):
    """ 以多进程的方式开启gearman的worker """
    try:
        # 记录子进程pid以便主进程被supervisor重启后清除上次未退出的子进程
        with open(pid, 'a+') as f:
            f.write("%d%s" % (os.getpid(), os.linesep))

        print('Chile process start for work.')
        continue_working = True
        worker_connections = []
        d = threading.Thread(name='monitor', target=gearman_monitor,
                             args=(gm_worker.redis_host, gm_worker.redis_port))
        d.start()

        def continue_while_connections_alive(any_activity):
            return gm_worker.after_poll(any_activity)

        # Shuffle our connections after the poll timeout
        while continue_working and GEARMAN_CONTINUE_WORK:
            worker_connections = gm_worker.establish_worker_connections()
            continue_working = gm_worker.poll_connections_until_stopped(
                worker_connections, continue_while_connections_alive, timeout=poll_timeout)

        # If we were kicked out of the worker loop, we should shutdown all our connections
        for current_connection in worker_connections:
            current_connection.close()

        print('Gearman worker closed')
        return None
    except Exception as e:
        print(e)


def gearman_monitor(redis_host, redis_port):
    """ 监听动态更新指令 """
    global GEARMAN_CONTINUE_WORK
    print('Start gearman monitor.')
    while GEARMAN_CONTINUE_WORK:
        # 防止运行异常导致线程挂死后无法监听redis响应，异常处理放在此处，发生异常后重新监听
        try:
            sub = redis.StrictRedis(redis_host, redis_port).pubsub()
            sub.subscribe('hot')
            for i in sub.listen():
                if isinstance(i.get('data'), str):
                    if i.get('data') == 'exit':
                        # worker退出的过程中将无法响应其他数据修改请求
                        print('Gearman monitor receive restart signal.')
                        GEARMAN_CONTINUE_WORK = False
                        sub.unsubscribe('hot')
                        break
                # 因线程间变量共享,故此处可用于多进程gearman worker运行中数据的更改

        except Exception as e:
            print(e)
            try:
                sub.unsubscribe('hot')
            except Exception:
                pass

    print('Gearman monitor closed')


if __name__ == '__main__':
    def test_multi_gearman_worker(worker, job):
        print(worker)
        print(job)

    gearman_worker = MultiGearmanWorker(('127.0.0.1:4730', ), '127.0.0.1', 6379)
    gearman_worker.register_task('test_multi_gearman_worker', test_multi_gearman_worker)
    gearman_worker.work()
