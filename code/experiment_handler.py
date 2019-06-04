
from queue import Queue
import datetime
from threading import Thread, Event
import numpy as np
from utils.ddf_logging import Logger
from utils.timer import Timer
from data_loader import DataLoader
import pickle
import git
import traceback
import torch
import os

class ExperimentHandler:
    """
    Capsulating class responsible for training and training state.
    This includes file disk I/O for intermediate states.
    Should be used in a 'with' environment, but can be reused.
    This is to correctly handle the DataThread and Logger attached.
    The constructor takes a config dictionary as sole argument.
    """

    class DataThread(Thread):
        """
        Internal class to handle data loading in a separate thread.
        """

        def __init__(self, data_loader, logger):
            super().__init__(name="Data Loader")
            self.queue = Queue(maxsize=8)
            self._run = Event()
            self._kill_pill = Event()
            self._data_loader = data_loader
            self._current_phase = None
            self._logger = logger

        def run(self):
            try:
                while self._run.wait():
                    if self._kill_pill.is_set():
                        break
                    self._data_loader.initialize_phase(self._current_phase)
                    for minibatch_in, minibatch_target in self._data_loader:
                        self.queue.put([minibatch_in, minibatch_target])
                    self.queue.put(None)
                    if self._kill_pill.is_set():
                        break
                    self._run.clear()
            except Exception as e:
                self._logger.print("Error: {0}".format(e))
                self._logger.print(traceback.format_exc())


        def serve(self, phase):
            """
            Order the DataThread to serve one epoch's worth of elements to the queue.
            The end-of-epoch is signaled by serving None.
            """
            self._current_phase = phase
            self._run.set()


        def kill(self):
            """
            Signal this thread that it is time to die.
            This signal can only be picked up if the thread is not serving at that point.
            """
            self._kill_pill.set()
            self._run.set()


    def get(self, key):
        "Look up a setting in the config dictionary."
        return self._config.get(key, None)


    def set(self, key, value):
        "Update a setting in the config dictionary."
        self._config[key] = value


    def __init__(self, config):
        self._config = config
        "The configuration of the network."

        if self.get('epochs_trained') is None or self.get('epochs_trained') == 0:
            self.set('epochs_trained', 0)
                
            self.network = self.get('network')(**self.get('network_options'))
            "torch.nn.Module responsible for mapping from input to output."

            self.optimizer = self.get('optimizer')(
                self.network.parameters(),
                **self.get('optimizer_options')
            )
            "nn.optim.Optimizer to use during training."

            self.losses = {
                'training_losses': [],
                'testing_losses': [],
                'testing_epochs': [],
            }
            "The internal buffer for loss evolution."

            self._logger = None
        else:
            self.output_path = os.path.join(self.get('output_folder'), self.get_experiment_identifier())
            "The path to load this experiment's output from."

            self._logger = Logger(self.output_path, self.get('console_verbosity'), self.get('log_to_file'))
            "The logger responsible for saving all information."

            with self._logger:
                self.load_state_from_file()
                self._config['output_folder'] = self.output_path

        self._data_loader = DataLoader(
            self.get('data_adapter')(**self.get('data_adapter_options')),
            **self.get('data_loader_options')
        )
        "DataLoader to interface with for getting data. Internal."
        self.data_loader_thread = None
        "DataThread responsible for serving (input,output) pairs."
        
        self.output_path = os.path.join(self.get('output_folder'), self.get_experiment_identifier())
        "The path to save this experiment's output."

        self.loss_fcn = self.get('loss_function')(**self.get('loss_function_options'))
        "loss function (should average over minibatch elements)."

        if self._logger is None:
            self._logger = Logger(self.output_path, self.get('console_verbosity'), self.get('log_to_file'))
            "The logger responsible for saving all information."

        self._data_loader.set_logger(self._logger)

        self._lr_milestones = self.get('optimizer_lr_milestones')
        """
        Dictionary (epoch, learning_rate) to control the learning rate over time.
        The learning rate will be set to this value AFTER that epoch.
        """


        "Some buffers defined here for memory efficiency"
        self.last_test_in = None
        self.last_test_out = None
        self.last_test_gt = None
        self.last_train_in = None
        self.last_train_out = None
        self.last_train_gt = None



    def get_statefile_name(self, epochs_trained=None):
        "Returns a unique filename for the current state."
        if epochs_trained is None:
            epochs_trained = self.get('epochs_trained')
        return os.path.join(self.output_path, "experiment_state_epoch_%05d.pkl" % epochs_trained)


    def save_state_to_file(self):
        "Pickle the current state to file."
        filename = self.get_statefile_name()
        os.makedirs(self.output_path, exist_ok=True)
        with open(filename,'wb') as file:
            pickle.dump(self._config, file)
            pickle.dump(self.losses, file)
            pickle.dump(self.network, file)
            pickle.dump(self.optimizer.state_dict(), file)


    def load_state_from_file(self, epoch=None):
        """
        Load the state for a given number of epochs from file.
        By default, loads the current epoch from file.
        """
        filename = self.get_statefile_name(epoch)

        # if there's a stored unpickler, use that one
        unpickler = self._config.pop('unpickler', None)
        if unpickler is not None:
            unpickle = lambda x: unpickler(x).load()
        else:
            unpickle = lambda x: pickle.load(x)

        with open(filename, 'rb') as file:
            self._config = unpickle(file)
            self.losses = unpickle(file)
            self.network = unpickle(file)
            optimizer_state = unpickle(file)

        self.optimizer = self.get('optimizer')(
            self.network.parameters(),
            **self.get('optimizer_options')
        )
        try:
            self.optimizer.load_state_dict(optimizer_state)
        except ValueError:
            self._logger.print(
                "[Warning] Could not load the optimizer state!",
                Logger.MESSAGE_WARNING
            )


    @staticmethod
    def load_experiment_from_file(filename, folder_override=None, unpickler=None):
        """
        Load an experiment, with all of its state and config, from a file.
        No safeguards that the pointed-at file is a valid experiment dump!
        All we do is load the relevant config dump, and then let the
        constructor take over.
        """

        if unpickler is not None:
            unpickle = lambda x: unpickler(x).load()
        else:
            unpickle = lambda x: pickle.load(x)

        with open(filename, 'rb') as file:
            config = unpickle(file)
        if folder_override is not None:
            config['output_folder'] = folder_override

        if unpickler is not None:
            config['unpickler'] = unpickler

        return ExperimentHandler(config)


    def __enter__(self):
        self.data_loader_thread = self.DataThread(self._data_loader, self._logger)
        self.data_loader_thread.start()
        self._logger.__enter__()
        return self


    def __exit__(self, *args):
        self._logger.__exit__(*args)
        self.data_loader_thread.kill()
        self.data_loader_thread.join()


    def run(self):
        "Sets up the environment, and runs the code in config.experiment."
        with self as handler:
            try:
                self.get('setup')(handler)
                self.get('experiment')(handler)
            except Exception as e:
                self._logger.print("Error: {0}".format(e))
                self._logger.print(traceback.format_exc())
                raise

    def get_experiment_identifier(self):
        """
        Construct an informative name for this experiment.
        Uses get_data_name and get_network_name from the data_loader and the network, respectively.
        """
        identifier = self.get('experiment_identifier')
        if identifier is None:
            experiment_name = self.get('experiment_name')
            if experiment_name is None:
                experiment_name = "%s_%s" % (
                    self._data_loader.get_data_name(),
                    self.network.get_network_name(),
                )

            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha

            identifier = "%s/%s-%s" % (
                experiment_name,
                datetime.datetime.now().strftime("%Y%m%d-%H%M"),
                repo.git.rev_parse(sha, short=6)
            )
            self.set('experiment_identifier', identifier)
        return identifier


    def train(self, nr_epochs=1, optimizer=None, print_prefix=None, direction=1.0):
        """
        Perform a given number of training epochs.
        """

        if nr_epochs == 0:
            return None

        if optimizer is None:
            optimizer = self.optimizer

        for _i in range(nr_epochs):
            self.data_loader_thread.serve('train')
            loss = None
            nr_elements = 0
            epoch_finished = False
            while not epoch_finished:
                data = self.data_loader_thread.queue.get()
                self.data_loader_thread.queue.task_done()
                if data is None:
                    epoch_finished = True
                    break

                minibatch_out = self.network(*data[0])
                if optimizer != self.optimizer:
                    minibatch_out = minibatch_out.detach()
                minibatch_loss = self.loss_fcn(minibatch_out, *data[1])
                if loss is None:
                    loss = minibatch_loss.detach() * data[0][0].size(0)
                else:
                    loss += minibatch_loss.detach() * data[0][0].size(0)
                nr_elements += data[0][0].shape[0]

                optimizer.zero_grad()
                minibatch_loss = minibatch_loss * direction
                if minibatch_loss.numel() > 1:
                    minibatch_loss = minibatch_loss.sum()
                minibatch_loss.backward()
                optimizer.step()

                # clear these buffers 
                if self.last_train_in is not None:
                    del self.last_train_in
                    self.last_train_in = None
                if self.last_train_out is not None:
                    del self.last_train_out
                    self.last_train_out = None
                if self.last_train_gt is not None:
                    del self.last_train_gt
                    self.last_train_gt = None
                self.last_train_in = data[0]
                if isinstance(minibatch_out, (tuple, list)):
                    self.last_train_out = [e.detach() for e in minibatch_out]
                else:
                    self.last_train_out = minibatch_out.detach()
                self.last_train_gt = data[1]

                del data
                del minibatch_out
                del minibatch_loss
                

            loss = loss.detach() / nr_elements

            if optimizer == self.optimizer:
                self.set('epochs_trained', self.get('epochs_trained') + 1)
                self.losses['training_losses'].append(loss.detach().cpu().numpy())

                if self._lr_milestones is not None:
                    new_lr = self._lr_milestones.get(self.get('epochs_trained'))
                    if new_lr is not None:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr

            if loss.numel() > 1:
                loss = loss.sum()

            self._logger.print("Epoch %5d\t[Train%s]\tloss: %f" % (
                self.get('epochs_trained'),
                "" if print_prefix is None else " %s" % print_prefix,
                loss,
            ))


        return loss


    def test(self):
        """
        Perform a single testing epoch.
        Will not re-run testing for an epoch which was already tested.
        """
        testing_epochs = self.losses['testing_epochs']
        testing_losses = self.losses['testing_losses']
        if len(testing_epochs) > 0 and testing_epochs[-1] == self.get('epochs_trained'):
            return testing_losses[-1]

        self.data_loader_thread.serve('test')
        loss = None
        nr_elements = 0
        epoch_finished = False
        while not epoch_finished:
            data = self.data_loader_thread.queue.get()
            self.data_loader_thread.queue.task_done()
            if data is None:
                epoch_finished = True
                break

            with torch.no_grad():
                minibatch_out = self.network(*data[0])
                minibatch_loss = self.loss_fcn(minibatch_out, *data[1])

            if loss is None:
                loss = minibatch_loss.detach() * data[0][0].size(0)
            else:
                loss += minibatch_loss.detach() * data[0][0].size(0)
            nr_elements += data[0][0].shape[0]
            
            # clear these buffers 
            if self.last_test_in is not None:
                del self.last_test_in
                self.last_test_in = None
            if self.last_test_out is not None:
                del self.last_test_out
                self.last_test_out = None
            if self.last_test_gt is not None:
                del self.last_test_gt
                self.last_test_gt = None

            
            self.last_test_in = data[0]
            if isinstance(minibatch_out, (tuple, list)):
                self.last_test_out = [e.detach() for e in minibatch_out]
            else:
                self.last_test_out = minibatch_out.detach()
            self.last_test_gt = data[1]

            del data
            del minibatch_out
            del minibatch_loss

        loss = loss.detach() / nr_elements
        testing_losses.append(loss.detach().cpu().numpy())
        testing_epochs.append(self.get('epochs_trained'))

        if loss.numel() > 1:
            loss = loss.sum()

        self._logger.print("Epoch %5d\t[Test]\tloss: %f" % (self.get('epochs_trained'), loss))
        return loss


    def plot_losses(self, max_samples=100):
        """
        Plot all available losses.
        If we there is more than the allowed number of samples, we simply subsample to that amount.

        Arguments:
            max_samples -- the maximum number of points per curve (defaults to 100)
            average -- whether or not to smooth the curve when subsampling (defaults to True)
        """
        training_epochs = np.arange(1, self.get('epochs_trained') + 1)
        training_losses = np.array(self.losses['training_losses'])
        testing_epochs = np.array(self.losses['testing_epochs'])
        testing_losses = np.array(self.losses['testing_losses'])

        plots = [training_losses, testing_losses]
        labels = [training_epochs, testing_epochs]
        colors = ['tab:blue', 'tab:orange']
        alphas = [1, 1]
        legends = ["train", "test"]

        if training_epochs.size > max_samples:
            step = training_epochs.size / max_samples
            samples = np.ceil(np.arange(step, training_epochs.size - step, step) - 1)
            samples = np.concatenate((np.array([0]),samples,np.array([training_epochs.size-1]))).astype(np.int)
            beta = 1-np.sqrt(max_samples/self.get('epochs_trained'))
            training_losses_ss = training_losses.copy()
            for i in range(1, training_epochs.size):
                training_losses_ss[i] = beta * training_losses_ss[i-1] + (1-beta) * training_losses_ss[i]
            training_epochs_ss = training_epochs[samples]
            training_losses_ss = training_losses_ss[samples]

            
            plots.append(training_losses_ss)
            labels.append(training_epochs_ss)
            colors.append(colors[0])
            alphas.append(1.0)
            alphas[0] = 0.3

        if testing_epochs.size > max_samples:
            step = testing_epochs.size / max_samples
            samples = np.ceil(np.arange(step, testing_epochs.size - step, step) - 1).astype(np.int)
            samples = np.concatenate((np.array([0]),samples,np.array([testing_epochs.size-1]))).astype(np.int)
            beta = 1-np.sqrt(max_samples/self.get('epochs_trained'))
            testing_losses_ss = testing_losses.copy()
            for i in range(1, testing_epochs.size):
                this_beta = np.power(beta, testing_epochs[i] - testing_epochs[i-1])
                testing_losses_ss[i] = this_beta * testing_losses_ss[i-1] + (1-this_beta) * testing_losses_ss[i]
            testing_epochs_ss = testing_epochs[samples]
            testing_losses_ss = testing_losses_ss[samples]

            plots.append(testing_losses_ss)
            labels.append(testing_epochs_ss)
            colors.append(colors[1])
            alphas.append(1.0)
            alphas[1] = 0.3

        self._logger.plot(
            plots,
            "Loss_evolution for %s - %s" % (self.network.get_network_name(), self.get('experiment_name')),
            xlabels=labels,
            xaxis="epochs",
            yaxis="loss",
            legends=legends,
            colors=colors,
            alphas=alphas
        )
