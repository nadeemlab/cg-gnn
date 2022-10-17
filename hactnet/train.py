#!/usr/bin/env python3
"""
Train CG-GNN models
"""
from os import makedirs, listdir
from os.path import exists, join
from shutil import rmtree
from typing import Callable, List, Tuple, Optional, Any, Sequence, Dict

from pandas.io.json import json_normalize
from mlflow import log_params, log_metric, log_artifact
from mlflow.pytorch import log_model
from torch import save, load, no_grad, argmax, cat
from torch.cuda import is_available
from torch.optim import Adam, Optimizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from dgl import DGLGraph
from tqdm import tqdm

from hactnet.histocartography.ml import CellGraphModel

from hactnet.util import CGDataset, collate

# cuda support
IS_CUDA = is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'

# model parameters
DEFAULT_GNN_PARAMS = {
    'layer_type': "pna_layer",
    'output_dim': 64,
    'num_layers': 3,
    'readout_op': "lstm",
    'readout_type': "mean",
    'aggregators': "mean max min std",
    'scalers': "identity amplification attenuation",
    'avg_d': 4,
    'dropout': 0.,
    'graph_norm': True,
    'batch_norm': True,
    'towers': 1,
    'pretrans_layers': 1,
    'posttrans_layers': 1,
    'divide_input': False,
    'residual': False
}
DEFAULT_CLASSIFICATION_PARAMS = {
    'num_layers': 2,
    'hidden_dim': 128
}


def _set_save_path(model_path: str) -> str:
    "Generate model path if we need to duplicate it and set path to save checkpoints."
    if exists(model_path):
        increment = 2
        while exists(model_path + f'_{increment}'):
            increment += 1
        model_path += f'_{increment}'
    makedirs(model_path, exist_ok=False)
    return model_path


def _create_dataset(cell_graphs: Optional[Tuple[List[DGLGraph], List[int]]],
                    in_ram: bool = True
                    ) -> Optional[CGDataset]:
    "Make a cell graph dataset."
    return CGDataset(cell_graphs, load_in_ram=in_ram) if (cell_graphs is not None) else None


def _create_datasets(
    cell_graph_sets: Tuple[Tuple[List[DGLGraph], List[int]],
                           Optional[Tuple[List[DGLGraph], List[int]]],
                           Optional[Tuple[List[DGLGraph], List[int]]]],
    in_ram: bool = True,
    k: int = 3
) -> Tuple[CGDataset, Optional[CGDataset], Optional[CGDataset], Optional[KFold]]:
    "Make the cell and/or tissue graph datasets and the k-fold if necessary."

    train_dataset = _create_dataset(cell_graph_sets[0], in_ram)
    assert train_dataset is not None
    val_dataset = _create_dataset(cell_graph_sets[1], in_ram)
    test_dataset = _create_dataset(cell_graph_sets[2], in_ram)

    if (k > 0) and (val_dataset is not None):
        # stack train and validation dataloaders if both exist and k-fold cross val is activated
        train_dataset = ConcatDataset((train_dataset, val_dataset))
    elif (k == 0) and (val_dataset is None):
        # set k to 3 if not provided and no validation data is provided
        k = 3
    kfold = KFold(n_splits=k, shuffle=True) if k > 0 else None

    return train_dataset, val_dataset, test_dataset, kfold


def _create_model(
        example_cg: DGLGraph,
        num_classes: int,
        gnn_params: Dict[str, Any] = DEFAULT_GNN_PARAMS,
        classification_params: Dict[str, Any] = DEFAULT_CLASSIFICATION_PARAMS
) -> CellGraphModel:
    "Declare model."
    return CellGraphModel(
        gnn_params=gnn_params,
        classification_params=classification_params,
        node_dim=example_cg.ndata['feat'].shape[1],
        num_classes=num_classes
    ).to(DEVICE)


def _create_training_dataloaders(train_ids: Optional[Sequence[int]],
                                 test_ids: Optional[Sequence[int]],
                                 train_dataset: CGDataset,
                                 val_dataset: Optional[CGDataset],
                                 batch_size: int
                                 ) -> Tuple[DataLoader, DataLoader]:
    "Determine whether to k-fold and then create dataloaders."
    if (train_ids is None) or (test_ids is None):
        if val_dataset is None:
            raise ValueError("val_dataset must exist.")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate
        )
    else:
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_subsampler,
            collate_fn=collate
        )
        val_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=test_subsampler,
            collate_fn=collate
        )

    return train_dataloader, val_dataloader


def _train_step(model: CellGraphModel,
                train_dataloader: DataLoader,
                loss_fn: Callable,
                optimizer: Optimizer,
                epoch: int,
                fold: int,
                step: int,
                logger: str = None
                ) -> Tuple[CellGraphModel, int]:
    "Train for 1 epoch/fold."

    model.train()
    for batch in tqdm(train_dataloader, desc=f'Epoch training {epoch}, fold {fold}', unit='batch'):

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
        if logger == 'mlflow':
            log_metric('train_loss', loss.item(), step=step)

        # 4. increment step
        step += 1

    return model, step


def _val_step(model: CellGraphModel,
              val_dataloader: DataLoader,
              loss_fn: Callable,
              model_path: str,
              epoch: int,
              fold: int,
              step: int,
              best_val_loss: float,
              best_val_accuracy: float,
              best_val_weighted_f1_score: float,
              logger: str = None
              ) -> CellGraphModel:
    "Run validation step."

    model.eval()
    all_val_logits = []
    all_val_labels = []
    for batch in tqdm(val_dataloader, desc=f'Epoch validation {epoch}, fold {fold}', unit='batch'):
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
    if logger == 'mlflow':
        log_metric('val_loss', loss, step=step)
    if loss < best_val_loss:
        best_val_loss = loss
        save(model.state_dict(), join(
            model_path, 'model_best_val_loss.pt'))

    # compute & store accuracy + model
    all_val_preds = all_val_preds.detach().numpy()
    all_val_labels = all_val_labels.detach().numpy()
    accuracy = accuracy_score(all_val_labels, all_val_preds)
    if logger == 'mlflow':
        log_metric('val_accuracy', accuracy, step=step)
    if accuracy > best_val_accuracy:
        best_val_accuracy = accuracy
        save(model.state_dict(), join(
            model_path, 'model_best_val_accuracy.pt'))

    # compute & store weighted f1-score + model
    weighted_f1_score = f1_score(
        all_val_labels, all_val_preds, average='weighted')
    if logger == 'mlflow':
        log_metric('val_weighted_f1_score',
                   weighted_f1_score, step=step)
    if weighted_f1_score > best_val_weighted_f1_score:
        best_val_weighted_f1_score = weighted_f1_score
        save(model.state_dict(), join(
            model_path, 'model_best_val_weighted_f1_score.pt'))

    print(f'Val loss {loss}')
    print(f'Val weighted F1 score {weighted_f1_score}')
    print(f'Val accuracy {accuracy}')

    return model


def _test_model(model: CellGraphModel,
                test_dataset: CGDataset,
                batch_size: int,
                loss_fn: Callable,
                model_path: str,
                step: int,
                logger: Optional[str] = None
                ) -> CellGraphModel:
    model.eval()
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate
    )

    max_acc = -1.
    max_acc_model_checkpoint = {}

    for metric in ['best_val_loss', 'best_val_accuracy', 'best_val_weighted_f1_score']:

        print(f'\n*** Start testing w/ {metric} model ***')

        model_name = [f for f in listdir(
            model_path) if f.endswith(".pt") and metric in f][0]
        checkpoint = load(join(model_path, model_name))
        model.load_state_dict(checkpoint)

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
        if logger == 'mlflow':
            log_metric('best_test_loss_' + metric, loss)

        # compute & store accuracy
        all_test_preds = all_test_preds.detach().numpy()
        all_test_labels = all_test_labels.detach().numpy()
        accuracy = accuracy_score(all_test_labels, all_test_preds)
        if logger == 'mlflow':
            log_metric('best_test_accuracy_' +
                       metric, accuracy, step=step)
        if accuracy > max_acc:
            max_acc = accuracy
            max_acc_model_checkpoint = checkpoint

        # compute & store weighted f1-score
        weighted_f1_score = f1_score(
            all_test_labels, all_test_preds, average='weighted')
        if logger == 'mlflow':
            log_metric('best_test_weighted_f1_score_' +
                       metric, weighted_f1_score, step=step)

        # compute and store classification report
        report = classification_report(
            all_test_labels, all_test_preds, digits=4)
        out_path = join(model_path, 'classification_report.txt')
        with open(out_path, "w", encoding='utf-8') as f:
            f.write(report)

        if logger == 'mlflow':
            artifact_path = f'evaluators/class_report_{metric}'
            log_artifact(out_path, artifact_path=artifact_path)

        # log MLflow models
        if logger == 'mlflow':
            log_model(model, 'model_' + metric)

        print(f'Test loss {loss}')
        print(f'Test weighted F1 score {weighted_f1_score}')
        print(f'Test accuracy {accuracy}')

    model.load_state_dict(max_acc_model_checkpoint)
    return model


def train(cell_graph_sets: Tuple[Tuple[List[DGLGraph], List[int]],
                                 Optional[Tuple[List[DGLGraph], List[int]]],
                                 Optional[Tuple[List[DGLGraph], List[int]]]],
          save_path: str,
          logger: Optional[str] = None,
          in_ram: bool = True,
          epochs: int = 10,
          learning_rate: float = 10e-3,
          batch_size: int = 1,
          k: int = 0,
          gnn_params: Dict[str, Any] = DEFAULT_GNN_PARAMS,
          classification_params: Dict[str, Any] = DEFAULT_CLASSIFICATION_PARAMS
          ) -> CellGraphModel:
    "Train CG-GNN."

    # log parameters to logger
    if logger == 'mlflow':
        log_params({
            'batch_size': batch_size
        })
        df = json_normalize(dict(gnn_params=gnn_params,
                                 classification_params=classification_params))
        rep = {"graph_building.": "", "model_params.": "",
               "gnn_params.": ""}  # replacement for shorter key names
        for i, j in rep.items():
            df.columns = df.columns.str.replace(i, j)
        flatten_config = df.to_dict(orient='records')[0]
        for key, val in flatten_config.items():
            log_params({key: str(val)})

    # set path to save checkpoints
    save_path = _set_save_path(save_path)

    # make datasets (train, validation & test)
    train_dataset, val_dataset, test_dataset, kfold = _create_datasets(
        cell_graph_sets, in_ram, k)

    # declare model
    model = _create_model(cell_graph_sets[0][0][0],
                          int(max(cell_graph_sets[0][1]))+1,
                          gnn_params=gnn_params,
                          classification_params=classification_params)

    # build optimizer
    optimizer = Adam(model.parameters(),
                     lr=learning_rate,
                     weight_decay=5e-4)

    # define loss function
    loss_fn = CrossEntropyLoss()

    # training loop
    step: int = 0
    best_val_loss: float = 10e5
    best_val_accuracy: float = 0.
    best_val_weighted_f1_score: float = 0.
    for epoch in range(epochs):

        folds: List[Tuple[Optional[Any], Optional[Any]]] = list(
            kfold.split(train_dataset)) if (kfold is not None) else [(None, None)]

        for fold, (train_ids, test_ids) in enumerate(folds):

            # Determine whether to k-fold and if so how
            train_dataloader, val_dataloader = _create_training_dataloaders(
                train_ids, test_ids, train_dataset, val_dataset, batch_size)

            # A.) train for 1 epoch
            model = model.to(DEVICE)
            model, step = _train_step(
                model, train_dataloader, loss_fn, optimizer, epoch, fold, step, logger)

            # B.) validate
            model = _val_step(model, val_dataloader, loss_fn, save_path, epoch, fold, step,
                              best_val_loss, best_val_accuracy, best_val_weighted_f1_score, logger)

    # testing loop
    if test_dataset is not None:
        model = _test_model(model, test_dataset, batch_size,
                            loss_fn, save_path, step, logger)

    if logger == 'mlflow':
        rmtree(save_path)

    return model


def infer(cell_graphs: Tuple[List[DGLGraph], List[int]],
          model_checkpoint_path: str,
          in_ram: bool = True,
          batch_size: int = 1,
          gnn_params: Dict[str, Any] = DEFAULT_GNN_PARAMS,
          classification_params: Dict[str, Any] = DEFAULT_CLASSIFICATION_PARAMS
          ) -> None:
    """
    Test CG-GNN.
    Args:
        args (Namespace): parsed arguments.
    """

    # make test data loader
    dataset = _create_dataset(cell_graphs, in_ram)
    assert dataset is not None
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)

    # declare model and load weights
    model = _create_model(cell_graphs[0][0],
                          int(max(cell_graphs[1]))+1,
                          gnn_params=gnn_params,
                          classification_params=classification_params)
    model.load_state_dict(load(model_checkpoint_path))
    model.eval()

    # print # of parameters
    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    # start testing
    all_test_logits = []
    all_test_labels = []
    for batch in tqdm(dataloader, desc='Testing', unit='batch'):
        labels = batch[-1]
        data = batch[:-1]
        with no_grad():
            logits = model(*data)
        all_test_logits.append(logits)
        all_test_labels.append(labels)

    all_test_logits = cat(all_test_logits).cpu()
    all_test_preds = argmax(all_test_logits, dim=1)
    all_test_labels = cat(all_test_labels).cpu()

    all_test_preds = all_test_preds.detach().numpy()
    all_test_labels = all_test_labels.detach().numpy()

    accuracy = accuracy_score(all_test_labels, all_test_preds)
    weighted_f1_score = f1_score(
        all_test_labels, all_test_preds, average='weighted')
    report = classification_report(all_test_labels, all_test_preds)

    print(f'Test weighted F1 score {weighted_f1_score}')
    print(f'Test accuracy {accuracy}')
    print(f'Test classification report {report}')
