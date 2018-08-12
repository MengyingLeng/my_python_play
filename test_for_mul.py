import multiprocessing #多进程
import threading #多线程


def job(q):
    res = 0
    for i in range(1000):
        res += i + i**2 + i**3
    q.put(res)


if __name__ == '__main__':

    q = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=job, args=(q,))
    p1.start()
    p1.join()
    p2 = multiprocessing.Process(target=job, args=(q,))
    p2.start()
    p2.join()
    p3 = multiprocessing.Process(target=job, args=(q,))
    p3.start()
    p3.join()
    res1 = q.get()
    res2 = q.get()
    res3 = q.get()
    print(res1+res2+res3)