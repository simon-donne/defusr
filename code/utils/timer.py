
import datetime

class Timer:
    """
    Class for timing 'with' blocks.
    Can be used without an attached logger, and will simply print to stdout.
    """
    def __init__(self, message="Timed operation", logger=None):
        self._text = message
        self._logger = logger
        self._tic = -1


    def set_message(self, message):
        self._text = message


    @staticmethod
    def current_time_millis():
        return round(datetime.datetime.utcnow().timestamp() * 1000)


    def __enter__(self):
        self._tic = self.current_time_millis()
        return self


    def __exit__(self, *args):
        if self._tic < 0:
            raise UserWarning("Unbalanced tic/toc.")
        toc = self.current_time_millis()
        if self._logger is not None:
            printer = self._logger.print
        else:
            printer = print
        printer("%s: %d ms" % (self._text, toc - self._tic))
