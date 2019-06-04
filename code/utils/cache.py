
import torch
from threading import Lock

GPU = torch.device('cuda')

class Cache:
    """
    Class to wrap function calls.
    Each instance keeps its own buffer.
    Threadsafe.
    """

    def __init__(self, enabled=True, gpu=False):
        self.cache = {}
        self._mutex = Lock()
        self.enabled = enabled
        self.gpu = gpu


    def get(self, fun, args):
        """The function that will cache calls as requested."""
        if not self.enabled:
            return fun(*args)
        self._mutex.acquire(blocking=True)
        result = self.cache.get((fun, args), None)
        status = "hit"
        if result is None:
            status = "miss"
            result = fun(*args)
            if self.gpu:
                if isinstance(result, (tuple, list)):
                    result = [x.to(GPU) for x in result]
                else:
                    result = result.to(GPU)
            self.cache[(fun, args)] = result
        self._mutex.release()
        return result


    def clear(self):
        self._mutex.acquire(blocking=True)
        self.cache = {}
        self._mutex.release()
