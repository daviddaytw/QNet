import tensorflow as tf
from utils.lr_finder import LRFinder, MaxStepStoppingWithLogging

class Trainer():
    def __init__(self, args, model, **kwargs):
        self.args = args
        self.model = model
        self.model.compile(**kwargs)

        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=( 'val_' + kwargs['metrics'][0] if 'metrics' in kwargs else 'val_loss'),
            patience=2,
            verbose=1,
            restore_best_weights=True
        )

    def train(self, train_data, test_data):
        args = self.args
        model = self.model

        mss_l = MaxStepStoppingWithLogging(max_steps=-1) # just logging
        lr_finder = LRFinder(
                        train_data,
                        args.batch_size,
                        window_size=int(args.lr_finder[0]),
                        max_steps=int(args.lr_finder[1]),
                        filename=args.lr_finder[2]
                    )

        callbacks = [
            mss_l,
            self.early_stopping,
        ]

        if args.lr <= 0:
            callbacks.append(lr_finder)

        print(model.summary())
        fitting = self.model.fit(
                    train_data,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    validation_data=test_data,
                    verbose=1,
                    callbacks=callbacks
                )

        fitting.history['batch'] = mss_l.history
        return fitting
    