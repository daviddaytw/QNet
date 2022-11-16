import argparse, os, json
import datasets, models

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Configure training arugments.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', '-d', default='stackoverflow', type=str,
                        help='Select the training or evaluation dataset (' + ','.join(datasets.list_dataset()) + ').')
    parser.add_argument('--model', '-m', default='resqnet', type=str,
                        help='Select the trainig model (' + ','.join(models.list_model()) + ')')
    parser.add_argument('--seq_len', '-ml', default='8', type=int,
                        help='Input length for the model.')
    parser.add_argument('--embed_size', '-ed', default='2', type=int,
                        help='Embedding size for each token.')
    parser.add_argument('--num_blocks', '-nb', default='1', type=int,
                        help='Number of mini-blocks in the model.')
    parser.add_argument('--batch_size', '-bs', default='128', type=int,
                        help='Number of samples per batch a node.')
    parser.add_argument('--lr', '-lr', default='3e-4', type=float,
                        help='The initial learning rate to cosine decay, 0 means use lr_finder')
    parser.add_argument('--lr_finder', default=['4', '400', './logs/lr_finder'], nargs=3,
                        metavar=('window_size max_steps verbose_filename', ':', 'int int str'),
                        help='LRFinder finding initial learning rate automatically, only applicable for --lr 0')
    parser.add_argument('--epochs','-e', default='10', type=int,
                        help='Number of training loops over all training data')
    parser.add_argument('--distributed_nodes','-dn', nargs='+', type=str,
                        help='Domain of all nodes')
    parser.add_argument('--distributed_node_index','-dni', type=int,
                        help='Index of current Node')
    args = parser.parse_args(args)

    return args

def solve_args(args = None, multi_worker_strategy: bool = False):
    args = parse_args(args)
    print('Configuration: ', args)

    # should set env before use `MultiWorkerStrategy` decorator
    if multi_worker_strategy and args.distributed_nodes:
        nodes = []
        for node in args.distributed_nodes:
            nodes.append(node + ':39763')

        os.environ['TF_CONFIG'] = json.dumps({
            'cluster': {
                'worker': nodes
            },
            'task': {'type': 'worker', 'index': args.distributed_node_index}
        })

    # Increase learning rate with global batch size.
    args.lr *= args.batch_size

    return args