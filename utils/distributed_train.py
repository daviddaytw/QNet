import tensorflow as tf

class MultiWorkerStrategy:
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    def __init__(self, func):
        self.func = func

    def __call__(self, train_or_all_data, test_data = None):
        train_or_all_data = train_or_all_data.with_options(self.options)

        with self.strategy.scope():
            print('Num Replicas In Sync: ', self.strategy.num_replicas_in_sync)

            if test_data is not None:
                test_data = test_data.with_options(self.options)
                self.func(train_or_all_data, test_data)
            else:
                self.func(train_or_all_data)

