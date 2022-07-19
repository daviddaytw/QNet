import tensorflow as tf

class MultiWorkerStrategy:
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    def __init__(self, func):
        self.func = func

    def __call__(self, train_data, test_data):
        train_data = train_data.with_options(self.options)
        test_data = test_data.with_options(self.options)

        with self.strategy.scope():
            print('Num Replicas In Sync: ', self.strategy.num_replicas_in_sync)
            self.func(train_data, test_data)
