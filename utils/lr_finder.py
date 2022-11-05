import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.ticker import FormatStrFormatter

class MaxStepStoppingWithLogging(tf.keras.callbacks.Callback):
    def __init__(self, max_steps: int=400, tqdm_prefix: str=None) -> None:
        self.max_steps = max_steps
        self.tqdm_prefix = tqdm_prefix
        self.history = {'lr': [], 'losses': [], 'val_losses': [], 'training_steps': []}

        self.tqdm = None
        if self.max_steps > 0 and self.tqdm_prefix is not None:
            self.tqdm = tqdm(desc=f'{self.tqdm_prefix}: logging loss', total=self.max_steps, miniters=1)

    def on_train_begin(self, logs={}):
        if not hasattr(self.model.optimizer, "lr") or not hasattr(self.model.optimizer.lr, "__call__"):
            raise ValueError('Optimizer must have a "lr" attribute and be callable.')

        tf.get_logger().setLevel('ERROR')

    def on_epoch_end(self, epoch, logs={}):
        self.history['val_losses'].append(logs.get('loss'))

    def on_batch_end(self, batch, logs={}):
        self.history['lr'].append(float(self.model.optimizer.lr(len(self.history['training_steps']))))
        self.history['losses'].append(logs.get('loss'))
        self.history['training_steps'].append(len(self.history['training_steps']))

        if self.max_steps > 0 and len(self.history['training_steps']) >= self.max_steps:
            self.model.stop_training = True

        if self.tqdm is not None:
            self.tqdm.update(1)

    def on_train_end(self, logs={}):
        tf.get_logger().setLevel('INFO')

        self.model.stop_training = False
        self.history['val_losses'] = np.repeat(
            self.history['val_losses'], len(self.history['losses']) // max(1, len(self.history['val_losses']))
        )[:len(self.history['losses'])].tolist()

    def savefig(self, filename, callback):
        # checking race condition
        if not np.all(np.diff(self.history['training_steps']) > 0):
            print("LRFinder: WARNING! training_steps should be non-descending, learning rate is messy!")

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        plt.figure(figsize=(18,6), dpi=100)
        ax = plt.axes()
        ax.grid()
        ax.plot(self.history['lr'], self.history['losses'], label="Training")
        ax.plot(self.history['lr'], self.history['val_losses'], label="Validation")
        ax.set_xticks(np.linspace(min(self.history['lr']), max(self.history['lr']), 65))
        ax.set_xticklabels(np.linspace(min(self.history['lr']), max(self.history['lr']), 65), rotation=45, ha='right', rotation_mode='anchor')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.8f'))
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Losses")
        callback(ax)
        ax.legend()
        plt.savefig(filename)

        return ax

    def save_history(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df = pd.DataFrame(self.history).astype({'losses': 'float', 'val_losses': 'float', 'lr': 'float', 'training_steps': 'int'})
        df.to_csv(filename + '.csv')

class LRFinder(tf.keras.callbacks.Callback):
    def __init__(self, data, batch_size, base_lr=1e-6, window_size=4, max_steps: int=400, decay_rate: float=1.024, filename=None) -> None:
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.window_size = window_size
        self.max_steps = max_steps
        self.decay_rate = decay_rate
        self.filename = filename

    def on_train_begin(self, logs={}):
        if not hasattr(self.model.optimizer, 'learning_rate') or not hasattr(self.model.optimizer.learning_rate, 'initial_learning_rate'):
            raise ValueError('Optimizer must have a "learning_rate" attribute and it has "initial_learning_rate"')

        lr_decayed_fn = tf.keras.optimizers.schedules.ExponentialDecay(self.base_lr, decay_steps=1, decay_rate=self.decay_rate)
        print('LRFinder: Finding `best_base_lr` range from: [{:.0e}({:.8f}), {:.0e}({:.8f})]...'.format(
                lr_decayed_fn(0), lr_decayed_fn(0),
                lr_decayed_fn(self.max_steps), lr_decayed_fn(self.max_steps)
            )
        )

        # do not change optimizer directly as it would raise the variable has been deleted FailedPreconditionError
        origin_lr = self.model.optimizer.learning_rate
        self.model.optimizer.learning_rate = lr_decayed_fn

        self.mss_l = MaxStepStoppingWithLogging(max_steps=self.max_steps, tqdm_prefix='LRFinder')
        self.model.fit(
            self.data,
            batch_size=self.batch_size,
            epochs=1<<10,
            verbose=0,
            callbacks=[self.mss_l]
        )

        self.history = self.mss_l.history
        right_padded_size = self.window_size - len(self.history['losses']) % self.window_size
        window_loesses = np.pad(self.history['losses'], (0, right_padded_size), 'edge') \
                           .reshape((-1, self.window_size)) \
                           .mean(axis=1)
        window_loesses_diff = np.diff(window_loesses)

        best_base_lr_index = min(self.max_steps - 1, (window_loesses_diff.argmin() * self.window_size))
        best_base_lr = self.history['lr'][best_base_lr_index]

        print('LRFinder: best_base_lr = {:.0e}({:.8f})'.format(best_base_lr, best_base_lr))
        if best_base_lr_index == (self.window_size // 2) + 2:
            print('LRFinder: WARNING! you should set a smaller `base_lr`')
        elif best_base_lr_index == self.max_steps - 1:
            print('LRFinder: WARNING! you should set a larger `max_steps` or `decay_rate`')
        if not np.allclose(lr_decayed_fn(np.arange(self.max_steps)), self.history['lr']):
            print("LRFinder: WARNING! replace optimizer failed, learning rate is incorrect")

        self.model.optimizer.learning_rate = origin_lr
        self.model.optimizer.learning_rate.initial_learning_rate = best_base_lr
        if np.allclose(self.model.optimizer.learning_rate(np.arange(self.max_steps)), self.history['lr']) \
            or not np.allclose(self.model.optimizer.learning_rate(np.arange(self.max_steps)), origin_lr(np.arange(self.max_steps))):
            print("LRFinder: WARNING! restore optimizer failed, learning rate is incorrect")

        self.best_base_lr, self.best_base_lr_index = best_base_lr, best_base_lr_index

        if self.filename is not None:
            self.mss_l.save_history(self.filename)
            self.mss_l.savefig(self.filename, lambda ax: ax.plot(
                self.best_base_lr, self.history['losses'][self.best_base_lr_index], '-ro', label='Best Base LR'
            ))