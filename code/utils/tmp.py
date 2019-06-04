
import os
import random
import string
import pathlib
import shutil

BASE_FOLDER = "/tmp/sdonne"

class TemporaryFolder():
    """
    Class for creating and cleaning up a temporary folder.
    Use in a 'with' statement to delete it afterwards.
    Use just the normal constructor to not delete it.
    The folders are in /tmp, so will disappear after reboot!
    Can be reused, and will reuse the same temporary folder.
    """

    @staticmethod
    def _random_name():
        """
        Generate a random filename.
        """
        return  ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))


    def __init__(self, name=None):
        if name is None:
            name = self._random_name()
        self.name = BASE_FOLDER + '/' + name + '/'
        self.__enter__()
        return


    def __enter__(self):
        pathlib.Path(self.name).mkdir(parents=True, exist_ok=True)
        return self


    def __exit__(self, *args):
        # it might have been removed or copied somewhere or whatever
        if os.path.exists(self.name):
            shutil.rmtree(self.name)


    def __str__(self):
        return self.name


    def tmp_file(self, extension=""):
        """
        Generate a random filename in this folder, with an optional extension.
        """
        return self.name + self._random_name() + extension
