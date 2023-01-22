import multiprocessing
import tensorflow as tf
print('Number of CPU count: ', multiprocessing.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(multiprocessing.cpu_count())

from models import count_params
from utils.args_parser import solve_args
args = solve_args(multi_worker_strategy=True)

import os, time, json, csv
from pathlib import Path

import numpy as np
from datasets import get_dataset
from trainers import get_trainer

def save_log(history, val_metric: str='val_loss'):
    logs = {
        'config': vars(args),
        'history': history,
    }

    if val_metric == 'val_loss':
        logs['best_acc'] = min(history[val_metric])
    else:
        logs['best_acc'] = max(history[val_metric])
    print('Best score: ', logs['best_acc'])

    record = {
        'Complete Time': time.ctime(),
        'Model':args.model,
        'Dataset':args.dataset,
        'Embed size':args.embed_size,
        'Num block': args.num_blocks,
        'Best Score': logs['best_acc'],
        'Params': count_params(args),
        'args': json.dumps(vars(args))
    }

    dir_path = os.path.dirname(os.path.realpath(__file__))
    record_csv = dir_path + f'/logs/0-Record.csv'
    Path(record_csv).touch()
    with open(record_csv, 'r') as f:
        rows = list(csv.DictReader(f))
    with open(record_csv, 'w') as f:
        writer = csv.DictWriter(f, record.keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        writer.writerow(record)
    print('0-Record.csv updated')

def main(args):
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

    dataset = get_dataset(args.dataset)
    trainer = get_trainer(dataset.getTask())
    fitting = trainer.train(args, dataset)

    if args.distributed_node_index == 0 or args.distributed_node_index == None:
        if dataset.getTask() == 'classification':
            if dataset.getOutputSize() > 2:
                save_log(fitting.history, 'val_categorical_accuracy')
            else:
                save_log(fitting.history, 'val_binary_accuracy')
        elif dataset.getTask() == 'ner':
            save_log(fitting.history, 'val_f1_score')
        else:
            save_log(fitting.history)

if __name__ == '__main__':
    main(args)
