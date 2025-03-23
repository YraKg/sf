#!/usr/bin/env bash

#python  train.py --exp_name preact_k256_e32_ni20_adv --data_name cifar10 --data_dir ./data/cifar10 --model_name preactresnet --max_epoch 100 --batch_size 256 --lr 0.05  --train_mode rand --patience 10 -k 256 --epsilon 32 --n_iters 20 --gpu 0 
#python  train.py --exp_name preact_k512_e16_ni50_trades --data_name cifar10 --data_dir ./data/cifar10 --model_name preactresnet --max_epoch 50 --batch_size 256 --lr 0.05 --train_loss trades --train_mode rand --patience 10 -k 512 --epsilon 16 --n_iters 50 --gpu 0 
#python  train.py --exp_name wrn_k512_e16_ni50_trades --data_name cifar10 --data_dir ./data/cifar10 --model_name wideresnet70_16 --max_epoch 50 --batch_size 256 --lr 0.05 --train_loss trades --train_mode rand --patience 10 -k 512 --epsilon 16 --n_iters 50 --gpu 0 


#python dynamic_train_epochs.py --exp_name dynamic_train_epochs_ni20_adv --data_name cifar10 --data_dir ./data/cifar10 --model_name preactresnet --max_epoch 100 --batch_size 256 --lr 0.05  --train_mode rand --patience 10 -k 256 --epsilon 32 --n_iters 20 --gpu 0 


#python dynamic_train_batched.py --exp_name dynamic2_train_batched_ni20_adv --data_name cifar10 --data_dir ./data/cifar10 --model_name preactresnet --max_epoch 100 --batch_size 256 --lr 0.05  --train_mode rand --patience 10 -k 256 --epsilon 32 --n_iters 20 --gpu 0

#python  train.py --exp_name preact_k256_e32_ni100_adv --data_name cifar10 --data_dir ./data/cifar10 --model_name preactresnet --max_epoch 100 --batch_size 256 --lr 0.05  --train_mode rand --patience 10 -k 256 --epsilon 32 --n_iters 100 --gpu 0 

#python train.py --exp_name wrn_k256_e32_ni100_adv --data_name cifar10 --data_dir ./data/cifar10 --model_name wideresnet70_16 --max_epoch 100 --batch_size 256 --lr 0.05  --train_mode rand --patience 10 -k 256 --epsilon 32 --n_iters 100 --gpu 0

# python dynamic_train_img.py --exp_name preact_dynamic_img_adv_ni100 --data_name cifar10 --data_dir ./data/cifar10 --model_name preactresnet --max_epoch 100 --batch_size 256 --lr 0.05 --train_mode rand --patience 10 -k 256 --epsilon 32 --n_iters 100 --gpu 0 

#python train.py --exp_name FT_wrn34_20_k256_e32_ni100_trades --data_name cifar10  --data_dir ./data/cifar10 --model_name FT_wrn34_20 --max_epoch 6 --save_freq 1 --batch_size 100 --lr 0.05 --train_loss trades --train_mode rand --patience 10 -k 256 --epsilon 32 --n_iters 100 --gpu 0 

#python dynamic_train_img.py --exp_name  FT_wrn34_20_dynamic_img_adv_ni100 --data_name cifar10 --data_dir ./data/cifar10 --model_name FT_wrn34_20 --save_freq 1 --max_epoch 100 --batch_size 256 --lr 0.05 --train_mode rand --patience 10 -k 256 --epsilon 32 --n_iters 100 --gpu 0 

# python train.py --exp_name FT_wrn94_16_k256_e32_ni100_adv --data_name cifar10 --data_dir ./data/cifar10 --model_name FT_wrn94_16 --max_epoch 6 --save_freq 1 --batch_size 64 --lr 0.05  --train_mode rand --patience 10 -k 256 --epsilon 32 --n_iters 100 --gpu 0 

#python train.py --exp_name FT_wrn70_16_k256_e32_ni100_adv --data_name cifar10 --data_dir ./data/cifar10 --model_name FT_wrn70_16 --max_epoch 6 --save_freq 1 --batch_size 256 --lr 0.05  --train_mode rand --patience 10 -k 256 --epsilon 32 --n_iters 100 --gpu 0 

python train.py --exp_name  FT_swin-L_adv_ni50_v2  --data_name imagenet100 --data_dir ./data/imagenet100 --model_name FT_swin-L --save_freq 1 --print_freq 50 --max_epoch 100 --batch_size 48 --lr 0.025 --train_mode rand --patience 10 -k 12544 --epsilon 16 --n_iters 50 --gpu 0 
