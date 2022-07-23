import argparse, os, json

def parse_args():
    parser = argparse.ArgumentParser(description='Configure training arugments.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', '-d', default='stackoverflow', type=str,
                        help='Select the training dataset.')
    parser.add_argument('--model', '-m', default='qnet', type=str,
                        help='Select the trainig model (transformer, qnet, fnet)')
    parser.add_argument('--seq_len', '-ml', default='8', type=int,
                        help='Input length for the model.')
    parser.add_argument('--embed_size', '-ed', default='2', type=int,
                        help='Embedding size for each token.')
    parser.add_argument('--num_blocks', '-nb', default='1', type=int,
                        help='Number of mini-blocks in the model.')
    parser.add_argument('--qnet_depth', default=1, type=int,
                        help='Number of QNet blocks on the quantum computer, only appliable for QNet.')
    parser.add_argument('--batch_size', '-bs', default='128', type=int,
                        help='Number of samples per batch a node.')
    parser.add_argument('--lr', '-lr', default='3e-4', type=float,
                        help='The initial learning rate.')
    parser.add_argument('--epochs','-e', default='5', type=int,
                        help='Number of training loops over all training data')
    parser.add_argument('--distributed_nodes','-dn', nargs='+', type=str,
                        help='Domain of all nodes')
    parser.add_argument('--distributed_node_index','-dni', type=int,
                        help='Index of current Node')
    args = parser.parse_args()

    return args

def solve_args(multi_worker_strategy: bool):
    args = parse_args()
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

        # Global batch size should be batch_per_worker * num_workers
        args.batch_size *= len(args.distributed_nodes)
    
    # Increase learning rate with global batch size.
    args.lr *= args.batch_size

    return args