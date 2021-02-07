from collections import deque
from typing import Dict
import numpy as np

class Transform():
    """base class definition for trackable and reversable transforms"""
    def __init__(self, max_op_cache:int=1000):
        """
        Args:
            max_op_cache (int, optional): allowed cache operation count. Defaults to 1000.
        """
        self.__ops_queue = deque([], maxlen=max_op_cache)
        self.__tracking = False

    @property
    def tracking(self): return self.__tracking

    def enable_tracking(self): self.__tracking = True

    def disable_tracking(self): self.__tracking = False

    def flush(self):
        # clear the queue
        self.__ops_queue.clear()

    def register_op(self, op:Dict):
        self.__ops_queue.append(op)

    def adjust(self, pred_boxes:np.ndarray, **kwargs) -> np.ndarray:
        return pred_boxes

    def _adjust(self, pred_boxes:np.ndarray) -> np.ndarray:
        op = self.__ops_queue.pop()
        return self.adjust(pred_boxes, **op)