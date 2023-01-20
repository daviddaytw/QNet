import tensorflow as tf
from utils.lr_finder import LRFinder

class Trainer():
    def __init__(self, args, model, monitor=None, **kwargs):
        self.args = args
        self.model = model
        self.model.compile(**kwargs)

        if monitor is None:
            if 'metrics' in kwargs:
                monitor = 'val_' + str(kwargs['metrics'][0])
            else:
                monitor = 'val_loss'

        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=2,
            verbose=1,
            restore_best_weights=True
        )

    def train(self, train_data, test_data, callbacks=[]):
        args = self.args
        model = self.model

        model_callbacks = [
            self.early_stopping,
            tf.keras.callbacks.TensorBoard(update_freq=1),
            *callbacks,
        ]

        if args.lr <= 0:
            model_callbacks.append(
                LRFinder(
                        train_data,
                        args.batch_size,
                        window_size=int(args.lr_finder[0]),
                        max_steps=int(args.lr_finder[1]),
                        filename=args.lr_finder[2]
                )
            )

        print(model.summary())
        fitting = self.model.fit(
                    train_data.repeat(),
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    steps_per_epoch=args.steps_per_epoch,
                    validation_data=test_data,
                    validation_steps=int(args.steps_per_epoch * 0.1),
                    shuffle=True,
                    verbose=1,
                    callbacks=model_callbacks
                )

        for cb in model_callbacks:
            if hasattr(cb, 'history'):
                fitting.history[type(cb).__name__] = cb.history

        return fitting
