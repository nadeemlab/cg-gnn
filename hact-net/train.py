#!/usr/bin/env python3
"""
Script for training CG-GNN, TG-GNN and HACT models
"""
from os import makedirs, listdir
from os.path import exists, join
from uuid import uuid4
from shutil import rmtree
from argparse import ArgumentParser
from typing import List, Tuple, Optional, Any

from yaml import safe_load
from pandas.io.json import json_normalize
from mlflow import log_params, log_metric, log_artifact
from mlflow.pytorch import log_model
from torch import save, load, no_grad, argmax, cat
from torch.cuda import is_available
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from histocartography.ml import CellGraphModel, TissueGraphModel, HACTModel

from util import CGTGDataset, collate

# cuda support
IS_CUDA = is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'


def parse_arguments():
    "Parse command line arguments."
    parser = ArgumentParser()
    parser.add_argument(
        '--cg_path',
        type=str,
        help='path to the cell graphs.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--tg_path',
        type=str,
        help='path to tissue graphs.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--assign_mat_path',
        type=str,
        help='path to the assignment matrices.',
        default=None,
        required=False
    )
    parser.add_argument(
        '-conf',
        '--config_fpath',
        type=str,
        help='path to the config file.',
        default='',
        required=False
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='path to where the model is saved.',
        default='',
        required=False
    )
    parser.add_argument(
        '--model_name',
        type=str,
        help='name of the model.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--in_ram',
        help='if the data should be stored in RAM.',
        action='store_true',
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        help='batch size.',
        default=1,
        required=False
    )
    parser.add_argument(
        '--epochs', type=int, help='epochs.', default=10, required=False
    )
    parser.add_argument(
        '-l',
        '--learning_rate',
        type=float,
        help='learning rate.',
        default=10e-3,
        required=False
    )
    parser.add_argument(
        '--out_path',
        type=str,
        help='path to where the output data are saved (currently only for the interpretability).',
        default='../../data/graphs',
        required=False
    )
    parser.add_argument(
        '--logger',
        type=str,
        help='Logger type. Options are "mlflow" or "none"',
        required=False,
        default='none'
    )
    parser.add_argument(
        '--k',
        type=int,
        help='Folds to use in k-fold cross validation. 0 means don\'t use k-fold cross validation '
        'unless no validation dataset is provided, in which case k defaults to 3.',
        required=False,
        default=0
    )

    return parser.parse_args()


def main(args):
    """
    Train HACTNet, CG-GNN, or TG-GNN.
    Args:
        args (Namespace): parsed arguments.
    """

    # load config file
    with open(args.config_fpath, 'r', encoding='utf-8') as f:
        config = safe_load(f)

    # log parameters to logger
    if args.logger == 'mlflow':
        log_params({
            'batch_size': args.batch_size
        })
        df = json_normalize(config)
        rep = {"graph_building.": "", "model_params.": "",
               "gnn_params.": ""}  # replacement for shorter key names
        for i, j in rep.items():
            df.columns = df.columns.str.replace(i, j)
        flatten_config = df.to_dict(orient='records')[0]
        for key, val in flatten_config.items():
            log_params({key: str(val)})

    # set path to save checkpoints
    model_name = str(uuid4()) if (
        args.model_name is None) else args.model_name
    model_path = join(args.model_path, model_name)
    if exists(model_path):
        i = 2
        while exists(model_path + f'_{i}'):
            i += 1
        model_path += f'_{i}'
    makedirs(model_path, exist_ok=False)

    # make datasets (train, validation & test)
    test_cg_path = join(
        args.cg_path, 'test') if args.cg_path is not None else None
    test_tg_path = join(
        args.tg_path, 'test') if args.tg_path is not None else None
    test_dataset = CGTGDataset(
        cg_path=test_cg_path,
        tg_path=test_tg_path,
        assign_mat_path=join(
            args.assign_mat_path, 'test') if args.assign_mat_path is not None else None,
        load_in_ram=args.in_ram
    ) if (
        (test_cg_path is not None) and exists(test_cg_path) or
        (test_tg_path is not None) and exists(test_tg_path)
    ) else None

    val_cg_path = join(
        args.cg_path, 'val') if args.cg_path is not None else None
    val_tg_path = join(
        args.tg_path, 'val') if args.tg_path is not None else None
    val_dataset = CGTGDataset(
        cg_path=val_cg_path,
        tg_path=val_tg_path,
        assign_mat_path=join(
            args.assign_mat_path, 'val') if args.assign_mat_path is not None else None,
        load_in_ram=args.in_ram
    ) if (
        (val_cg_path is not None) and exists(val_cg_path) or
        (val_tg_path is not None) and exists(val_tg_path)
    ) else None

    train_dataset = CGTGDataset(
        cg_path=join(
            args.cg_path, 'train') if args.cg_path is not None else None,
        tg_path=join(
            args.tg_path, 'train') if args.tg_path is not None else None,
        assign_mat_path=join(
            args.assign_mat_path, 'train') if args.assign_mat_path is not None else None,
        load_in_ram=args.in_ram
    )

    k: int = args.k
    if (k > 0) and (val_dataset is not None):
        # stack train and validation dataloaders if both exist and k-fold cross val is activated
        train_dataset = ConcatDataset((train_dataset, val_dataset))
    elif (k == 0) and (val_dataset is None):
        # set k to 3 if not provided and no validation data is provided
        k = 3
    kfold = KFold(n_splits=k, shuffle=True) if k > 0 else None

    # declare model
    if 'cggnn' in args.config_fpath:
        model = CellGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=config['node_feat_dim'],
            num_classes=config['num_classes']
        ).to(DEVICE)
    elif 'tggnn' in args.config_fpath:
        model = TissueGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=config['node_feat_dim'],
            num_classes=config['num_classes']
        ).to(DEVICE)
    elif 'hact' in args.config_fpath:
        model = HACTModel(
            cg_gnn_params=config['cg_gnn_params'],
            tg_gnn_params=config['tg_gnn_params'],
            classification_params=config['classification_params'],
            cg_node_dim=config['node_feat_dim'],
            tg_node_dim=config['node_feat_dim'],
            num_classes=config['num_classes']
        ).to(DEVICE)
    else:
        raise ValueError(
            'Model type not recognized. Options are: TG, CG or HACT.')

    # build optimizer
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=5e-4
    )

    # define loss function
    loss_fn = CrossEntropyLoss()

    # training loop
    step = 0
    best_val_loss = 10e5
    best_val_accuracy = 0.
    best_val_weighted_f1_score = 0.
    epoch = 1
    while epoch < args.epochs + 1:

        folds: List[Tuple[Optional[Any], Optional[Any]]] = list(
            kfold.split(train_dataset)) if (kfold is not None) else [(None, None)]

        for train_ids, test_ids in folds:

            # Determine whether to k-fold and if so how
            if (train_ids is None) or (test_ids is None):
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=collate
                )
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=collate
                )
            else:
                train_subsampler = SubsetRandomSampler(train_ids)
                test_subsampler = SubsetRandomSampler(test_ids)
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    sampler=train_subsampler,
                    collate_fn=collate
                )
                val_dataloader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    sampler=test_subsampler,
                    collate_fn=collate
                )

            # A.) train for 1 epoch
            model = model.to(DEVICE)
            model.train()
            for batch in tqdm(train_dataloader, desc=f'Epoch training {epoch}', unit='batch'):

                # 1. forward pass
                labels = batch[-1]
                data = batch[:-1]
                logits = model(*data)

                # 2. backward pass
                loss = loss_fn(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 3. log training loss
                if args.logger == 'mlflow':
                    log_metric('train_loss', loss.item(), step=step)

                # 4. increment step
                step += 1

            # B.) validate
            model.eval()
            all_val_logits = []
            all_val_labels = []
            for batch in tqdm(val_dataloader, desc=f'Epoch validation {epoch}', unit='batch'):
                labels = batch[-1]
                data = batch[:-1]
                with no_grad():
                    logits = model(*data)
                all_val_logits.append(logits)
                all_val_labels.append(labels)

            all_val_logits = cat(all_val_logits).cpu()
            all_val_preds = argmax(all_val_logits, dim=1)
            all_val_labels = cat(all_val_labels).cpu()

            # compute & store loss + model
            with no_grad():
                loss = loss_fn(all_val_logits, all_val_labels).item()
            if args.logger == 'mlflow':
                log_metric('val_loss', loss, step=step)
            if loss < best_val_loss:
                best_val_loss = loss
                save(model.state_dict(), join(
                    model_path, 'model_best_val_loss.pt'))

            # compute & store accuracy + model
            all_val_preds = all_val_preds.detach().numpy()
            all_val_labels = all_val_labels.detach().numpy()
            accuracy = accuracy_score(all_val_labels, all_val_preds)
            if args.logger == 'mlflow':
                log_metric('val_accuracy', accuracy, step=step)
            if accuracy > best_val_accuracy:
                best_val_accuracy = accuracy
                save(model.state_dict(), join(
                    model_path, 'model_best_val_accuracy.pt'))

            # compute & store weighted f1-score + model
            weighted_f1_score = f1_score(
                all_val_labels, all_val_preds, average='weighted')
            if args.logger == 'mlflow':
                log_metric('val_weighted_f1_score',
                           weighted_f1_score, step=step)
            if weighted_f1_score > best_val_weighted_f1_score:
                best_val_weighted_f1_score = weighted_f1_score
                save(model.state_dict(), join(
                    model_path, 'model_best_val_weighted_f1_score.pt'))

            print(f'Val loss {loss}')
            print(f'Val weighted F1 score {weighted_f1_score}')
            print(f'Val accuracy {accuracy}')

            # Increment epoch, counting each fold as an epoch
            epoch += 1

    # testing loop
    model.eval()
    if test_dataset is not None:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate
        )

        for metric in ['best_val_loss', 'best_val_accuracy', 'best_val_weighted_f1_score']:

            print(f'\n*** Start testing w/ {metric} model ***')

            model_name = [f for f in listdir(
                model_path) if f.endswith(".pt") and metric in f][0]
            model.load_state_dict(load(join(model_path, model_name)))

            all_test_logits = []
            all_test_labels = []
            for batch in tqdm(test_dataloader, desc=f'Testing: {metric}', unit='batch'):
                labels = batch[-1]
                data = batch[:-1]
                with no_grad():
                    logits = model(*data)
                all_test_logits.append(logits)
                all_test_labels.append(labels)

            all_test_logits = cat(all_test_logits).cpu()
            all_test_preds = argmax(all_test_logits, dim=1)
            all_test_labels = cat(all_test_labels).cpu()

            # compute & store loss
            with no_grad():
                loss = loss_fn(all_test_logits, all_test_labels).item()
            if args.logger == 'mlflow':
                log_metric('best_test_loss_' + metric, loss)

            # compute & store accuracy
            all_test_preds = all_test_preds.detach().numpy()
            all_test_labels = all_test_labels.detach().numpy()
            accuracy = accuracy_score(all_test_labels, all_test_preds)
            if args.logger == 'mlflow':
                log_metric('best_test_accuracy_' +
                           metric, accuracy, step=step)

            # compute & store weighted f1-score
            weighted_f1_score = f1_score(
                all_test_labels, all_test_preds, average='weighted')
            if args.logger == 'mlflow':
                log_metric('best_test_weighted_f1_score_' +
                           metric, weighted_f1_score, step=step)

            # compute and store classification report
            report = classification_report(
                all_test_labels, all_test_preds, digits=4)
            out_path = join(model_path, 'classification_report.txt')
            with open(out_path, "w", encoding='utf-8') as f:
                f.write(report)

            if args.logger == 'mlflow':
                artifact_path = f'evaluators/class_report_{metric}'
                log_artifact(out_path, artifact_path=artifact_path)

            # log MLflow models
            log_model(model, 'model_' + metric)

            print(f'Test loss {loss}')
            print(f'Test weighted F1 score {weighted_f1_score}')
            print(f'Test accuracy {accuracy}')

    if args.logger == 'mlflow':
        rmtree(model_path)


if __name__ == "__main__":
    main(args=parse_arguments())
