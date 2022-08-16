# QNet

This is the offical implementation of the paper "QNet: A Quantum-native Transformer Encoder Architecture".

## Training model

To train a new model, run the trainer with customized configuration.
```sh
python train.py [-h] [--dataset DATASET] [--model MODEL] [--seq_len SEQ_LEN]
                [--embed_size EMBED_SIZE] [--num_blocks NUM_BLOCKS]
                [--qnet_depth QNET_DEPTH] [--batch_size BATCH_SIZE] [--lr LR]
                [--epochs EPOCHS]
```

Configurable training arguments:
| options                 | explanation                                                                          |
|-------------------------|--------------------------------------------------------------------------------------|
| -h                      | show help message                                                                    |
| -d DATASET              | Select the training dataset. (default: stackoverflow)                                |
| -m MODEL                | Select the trainig model (transformer, qnet, fnet) (default: qnet)                   |
| -ml SEQ_LEN             | Input length for the model. (default: 8)                                             |
| -ed EMBED_SIZE          | Embedding size for each token. (default: 2)                                          |
| -nb NUM_BLOCKS          | Number of mini-blocks in the model. (default: 1)                                     |
| --qnet_depth QNET_DEPTH | Number of QNet blocks on the quantum computer, only appliable for QNet. (default: 1) |
| -bs BATCH_SIZE          | Number of samples per batch a node. (default: 128)                                   |
| -lr LR                  | The initial learning rate. (default: 3e-4)                                           |
| -e EPOCHS               | Number of training loops over all training data. (default: 5)                        |

### Distributed Training

The script to train the model in the distributed environment is in `scripts/` directory.

Configurable arguments for distributed training:
| options                                 | explanation                                |
|-----------------------------------------|--------------------------------------------|
| --num_nodes NUM_NODES                   | The number of computing nodes.             |
| --pbs_log_file LOG_FILE                 | The path to store the log file.            |
| --notification_email NOTIFICATION_EMAIL | The job status will be sent to this email. |
