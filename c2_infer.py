#!/usr/bin/env python3
"""
Script for testing CG-GNN, TG-GNN and HACT models
"""

from argparse import ArgumentParser

from hactnet.train import infer
from hactnet.util import load_cgtg_graphs


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
        help='path to model to test.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--in_ram',
        help='if the data should be stored in RAM.',
        action='store_true',
    )
    parser.add_argument(
        '--pretrained',
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    infer(args.config_fpath,
          args.model_path,
          load_cgtg_graphs(args.cg_path) if args.cg_path is not None else None,
          load_cgtg_graphs(args.tg_path) if args.tg_path is not None else None,
          args.assign_mat_path,
          args.in_ram,
          args.batch_size,
          args.pretrained)
