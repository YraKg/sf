from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse

def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='Train adversal attack network')
    parser.add_argument('--exp_name',           dest='exp_name', 
                        type=str,               default='debug', 
                        help='exp name used to construct output dir')
    parser.add_argument('--snap_dir',           dest='snap_dir', 
                        type=str,               default='snapshots', 
                        help='directory to save model')
    parser.add_argument('--log_dir',            dest='log_dir', 
                        type=str,               default='logs', 
                        help='directort to save logs')
    parser.add_argument('--no_log',             dest='no_log',
                        action='store_true',
                        help="if record logs (do not log)")
    # dataset and model settings
    parser.add_argument('--data_name',          dest='data_name', 
                        type=str,               default='mnist', 
                        help='used dataset')
    parser.add_argument('--data_dir',           dest='data_dir', 
                        type=str,               default='data/mnist', 
                        help='data directory')

    parser.add_argument('--model_name',         dest='model_name', 
                        type=str,               default='mnist', 
                        help='network model')
    parser.add_argument('--ckpt_path', type=str)
    # training settings
    parser.add_argument('--max_epoch',          dest='max_epoch', 
                        type=int,               default=100, 
                        help='max train steps')
    parser.add_argument('--lr',                 dest='lr', 
                        type=float,             default=0.05,
                        help='learning rate')
    parser.add_argument('--batch_size',         dest='batch_size', 
                        type=int,               default=128,
                        help='training batch size')
    parser.add_argument('--seed',               dest='seed', 
                        type=int,               default=0, 
                        help='random seed')
    parser.add_argument('--gpu',                dest="gpus", 
                        type=str,               default="0",
                        help="GPU to be used, default is '0' ")
    parser.add_argument('--rand',               dest='randomize', 
                        action='store_true', 
                        help='randomize (not use a fixed seed)')
    parser.add_argument('--steps',              dest='steps', 
                        type=str,               default='50,75',
                        help='epoches to decrease learning rate')
    parser.add_argument('--decay_rate',         dest='decay_rate', 
                        type=float,             default=0.1, 
                        help='decay rate to decrease learning rate')
    parser.add_argument('--train_mode',                type=str,
                        default='rand', choices=['rand', 'unproj', 'proj', 'alter'],
                        help='train mode')

    # print and output settings
    parser.add_argument('--print_freq',         dest='print_freq', 
                        type=int,               default=10, 
                        help='print freq')
    parser.add_argument('--output_freq',        dest='output_freq', 
                        type=int,               default=5, 
                        help='output freq')
    parser.add_argument('--save_freq',          dest='save_freq', 
                        type=int,               default=5,
                        help='save checkpint freq')
    parser.add_argument('--smaller_train_set',          dest='smaller_train_set', 
                        type=bool,               default=False,
                        help='make the training set smaller')
    # evaluation settings
    parser.add_argument('--eval_model',         dest='eval_model', 
                        type=str,               default=None, 
                        help='evaluation checkpint path')
    parser.add_argument('--eval_samples',       dest='eval_samples', 
                        type=int,               default=10000, 
                        help='num of evaluation samples')
    parser.add_argument('--eval_batch_size',    dest='eval_batch_size', 
                        type=int,               default=256,
                        help='evaluation batch size')
    parser.add_argument('--eval_cpu',           dest='eval_cpu',
                        action='store_true',
                        help="if eval on cpu (do not on cpu)")
    # adversarial examples settings
    parser.add_argument('--epsilon',            dest='epsilon', 
                        type=float,             default=255,
                        help='the maximum allowed perturbation per pixel')
    parser.add_argument('-k',                  dest='k',
                        type=int,               default=20,
                        help='sparsity level')
    parser.add_argument('--n_iters', dest='n_iters',
                        type=int,               default=100,
                        help='number of iterations')
    parser.add_argument('--alpha',              dest='alpha', 
                        type=float,             default=0.25,
                        help='the step size of magnitude p')
    parser.add_argument('--beta',              dest='beta',
                        type=float,             default=0.25,
                        help='the step size of sparsity mask')
    parser.add_argument('--patience',              dest='patience',
                        type=int,             default=10,
                        help='the step size of sparsity mask')
    parser.add_argument('--train_loss',              dest='train_loss',
                        type=str,               default='adv',
                        choices=['adv', 'trades'],
                        help='epoches to decrease learning rate')
    parser.add_argument('--trades_beta', dest='trades_beta',
                        type=float, default=3,
                        help='scaling factor of robust loss in TRADES')
    args = parser.parse_args()
    return args

cfg = parse_args()
cfg.steps = cfg.steps.replace('[', '').replace(']', '')
if cfg.steps:
    cfg.steps = [int(x) for x in cfg.steps.split(',') if x.isdigit()]
else:
    cfg.steps = []
if cfg.epsilon > 1: cfg.epsilon /= 255.
if cfg.alpha > 1: cfg.alpha /= 255.
