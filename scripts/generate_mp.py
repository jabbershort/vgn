from time import sleep
from multiprocessing import Queue, Pool, Process, current_process, cpu_count

GRASPS_PER_SCENE = 120

queue = Queue()


class DataGenerator:
    def __init__(self,count):
        self.grasp_count = count
        self.num_workers = cpu_count()

    @property
    def scenes(self):
        return [*range(int(self.grasp_count/GRASPS_PER_SCENE)+1)]

    def worker_thread(self):
        while self.queue.qsize() >0:
            record = self.queue.get()
            print(f'Worker: {current_process().name}: {record}')
            # TODO: do work
            sleep(1)
        print(f'Worker: {current_process().name} finished')

    def run(self):
        self.queue = Queue()
        for id in self.scenes:
            self.queue.put(id)
        self.processes = [Process(target=self.worker_thread) for _ in range(self.num_workers)]

        for process in self.processes:
            process.start()
            print('Process started')

        for process in self.processes:
            process.join()      

def mp_worker(queue):

    while queue.qsize() >0 :
        record = queue.get()
        print(f'Worker: {current_process().name}: {record}')
        sleep(1)

    print("worker closed")

def mp_handler():

    # Spawn two processes, assigning the method to be executed 
    # and the input arguments (the queue)
    processes = [Process(target=mp_worker, args=(queue,)) for _ in range(cpu_count())]

    for process in processes:
        process.start()
        print('Process started')

    for process in processes:
        process.join()



if __name__ == '__main__':

    # for id in id_list:
    #     queue.put(id)

    # mp_handler()
    gen = DataGenerator(1000)
    gen.run()
