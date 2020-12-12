from queue import Queue
from typing import Dict
import numpy as np

class Transform():
    """base class definition for trackable and reversable transforms"""
    def __init__(self, max_op_cache:int=1000):
        """
        Args:
            max_op_cache (int, optional): allowed cache operation count. Defaults to 1000.
        """
        self.__ops_queue = Queue(maxsize=max_op_cache)
        self.__tracking = False

    @property
    def tracking(self): return self.__tracking

    def enable_tracking(self): self.__tracking = True

    def disable_tracking(self): self.__tracking = False

    def flush(self):
        self.__ops_queue.mutex.acquire()
        self.__ops_queue.queue.clear()
        self.__ops_queue.all_tasks_done.notify_all()
        self.__ops_queue.unfinished_tasks = 0
        self.__ops_queue.mutex.release()

    def register_op(self, op:Dict):
        self.__ops_queue.put_nowait(op)

    def adjust(self, pred_boxes:np.ndarray, **kwargs) -> np.ndarray:
        return pred_boxes

    def _adjust(self, pred_boxes:np.ndarray) -> np.ndarray:
        op = self.__ops_queue.get_nowait()
        return self.adjust(pred_boxes, **op)