"""
Central module for displaying and archiving run-time information.
For text, graphs, 3D model renders, and anything else.
As such, all visualization calls should go through an instance of this class.
"""

import pathlib
import torch
from termcolor import cprint
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cv2
from utils.visualization import plot
from utils.timer import Timer

class Logger:
    """
    Presents a unified interface between the calculation and presentation of results.
    Anything that is presented to the end user should pass through here.
    Can be used in 'with' statements for automatic closing of the relevant files.
    Includes tic/toc functionality.
    """

    MESSAGE_INFO = 1
    MESSAGE_WARNING = 2
    MESSAGE_ERROR = 3
    MESSAGE_NONE = 4

    _message_colors = {
        MESSAGE_INFO: 'white',
        MESSAGE_WARNING: 'yellow',
        MESSAGE_ERROR: 'red',
    }

    _message_prefixes = {
        MESSAGE_INFO: '[INFO]',
        MESSAGE_WARNING: '[WARNING]',
        MESSAGE_ERROR: '[RED]',
    }


    def __init__(self, log_path="", console_verbosity=MESSAGE_INFO, log_to_file=True):
        self.log_path = log_path
        "The root output directory for all things logs."
        self.console_verbosity = console_verbosity
        "How much is printed to the console. See MESSAGE_*."
        self._textlog = None
        "The file handle being written to. Only valid inside 'with' environments."
        self._log_to_file = log_to_file
        "Whether or not logs and images are written to the file system."
        self._tics = []
        "Internal buffer for timers."


    def __enter__(self):
        if self._log_to_file:
            pathlib.Path(self.log_path).mkdir(parents=True, exist_ok=True)
            self._textlog = open("%s/log.txt" % self.log_path, "a")
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        if self._log_to_file:
            self._textlog.close()


    def print(self, text, message_type=MESSAGE_INFO):
        """
        Print text to the console and write it to the log file.

        Arguments:
            text -- the text to print (string).
            [level] -- the type of message this is. See MESSAGE_*.
        """
        textcolor = self._message_colors.get(message_type, None)
        textprefix = self._message_prefixes.get(message_type, None)
        if textcolor is None or textprefix is None:
            self.print("[Logger] Message type %s is not known. \
                       See MESSAGE_* for supported types."
                       % message_type, self.MESSAGE_ERROR)
        if self.console_verbosity <= message_type:
            cprint(text, textcolor)
        if self._log_to_file:
            self._textlog.write("%s %s\n" % (textprefix, text))
            self._textlog.flush()


    def plot(self, data, name, **kwargs):
        """
        Plot torch data and saves it to file.
        Does not perform

        Arguments:
            data -- K rows of N-length data to plot (torch Tensor)
            name -- the title for this plot. Used in adapted form as filename.

        Keyword arguments:
            [xlabels] -- N-length indices on the x-axis (torch Tensor)
            [xaxis] -- name of the x axis (defaults to 'x')
            [yaxis] -- name of the x axis (defaults to 'y')
            [logscale_y] -- whether the y axis is drawn in log-scale. Defaults to True.
            [legend] -- legend for this figure (list of strings). Missing/None entries are ignored.
        """

        figure = plot(data, name, plot_to_screen=False, **kwargs)
        if self._log_to_file:
            figure.savefig("%s/%s.png" % (self.log_path, name.lower().split(' ')[0]))
        plt.close(figure)


    def imwrite(self, image, name):
        """
        Writes an image to the output directory.
        
        Arguments:
            image -- numpy array (H x W x C) or torch array (C x H x W)
            name -- name for the file (no extension)
        """
        if torch.is_tensor(image):
            image = image.numpy().transpose([1,2,0])
        cv2.imwrite("%s/%s.png" % (self.log_path, name), image)


    def tic(self):
        """
        Start a new timer and push it on the stack.
        """
        timer = Timer(logger=self)
        self._tics.append(timer)
        timer.__enter__()


    def toc(self, text):
        """
        Stop the most recently started timer on the stack.
        """
        if len(self._tics) == 0:
            raise UserWarning("Unbalanced tic/toc.")
        timer = self._tics.pop()
        timer.__exit__()
        timer.set_text(text)
