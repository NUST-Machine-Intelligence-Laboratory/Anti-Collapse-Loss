 #-------------------
 #   CUB200
 #-------------------
 ### BNInception, dim = 128
 # ProxyAnchor Baseline
# python main.py --seed 0 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
# --dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  bninception_frozen_normalize \
# --embed_dim 128 --loss oproxy --bs 90

 # ProxyAnchor Baseline
#CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
# --dataset cub200 --n_epochs 300 --tau 30 --gamma 0.1 --arch  bninception_frozen_normalize \
# --embed_dim 512 --loss nir_mcr2 --bs 90

#
# ## ProxyAnchor + NIR | DoublePool
# CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --loss_mcr2_w_com 0.004 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet\
# --dataset cub200 --n_epochs 140 --tau 140 --gamma 0.1 --arch  bninception_frozen_normalize_double \
# --embed_dim 512 --loss nir_mcr2 --bs 90 --warmup 40 --loss_nir_w_align 0.0075
#  CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --loss_mcr2_w_com 0.004 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet\
# --dataset cub200 --n_epochs 140 --tau 140 --gamma 0.1 --arch  bninception_frozen_normalize \
# --embed_dim 512 --loss nir_mcr2 --bs 90 --warmup 40 --loss_nir_w_align 0.0075
#
# ### ResNet50, dim = 512
# ## ProxyAnchor + NIR
# CUDA_VISIBLE_DEVICES=0 python main.py --seed 5 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
# --dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  resnet50_frozen_normalize \
# --embed_dim 512 --loss nir --bs 90 --warmup 1 --loss_nir_w_align 0.0075

# ## ProxyAnchor + NIR | DoublePool
# CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
# --dataset cub200 --n_epochs 450 --tau 60 --gamma 0.1 --arch  resnet50_frozen_normalize_double \
# --embed_dim 512 --loss nir_mcr2 --bs 90 --warmup 50 --loss_nir_w_align 0.0075

 # ## ProxyAnchor + NIR | DoublePool
# CUDA_VISIBLE_DEVICES=1 python main.py --seed 1 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
# --dataset cub200 --n_epochs 200 --tau 200 --gamma 0.1 --arch  resnet50_frozen_normalize_double \
# --embed_dim 512 --loss oproxy_mcr2 --bs 90 --warmup 1 --loss_nir_w_align 0.0075
#
#CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
# --dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  resnet50_frozen_normalize_double \
# --embed_dim 512  --loss nir_mcr2 --bs 90 --warmup 0 --samples_per_class 2 --loss_nir_w_align 0.0035


#CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
#--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  resnet50_frozen_normalize_double \
#--embed_dim 512  --loss nir_mcr2 --bs 90 --warmup 10 --samples_per_class 2 --loss_nir_w_align 0.0035
#
#CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
#--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  resnet50_frozen_normalize_double \
#--embed_dim 512  --loss nir_mcr2 --bs 90 --warmup 10 --samples_per_class 2 --loss_nir_w_align 0.0035
#
#CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
#--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  resnet50_frozen_normalize_double \
#--embed_dim 512  --loss nir_mcr2 --bs 90 --warmup 10 --samples_per_class 2 --loss_nir_w_align 0.0035
#
 #-------------------
 #   CUB200
 #-------------------
CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --no_train_metrics --source_path ../DataSet \
--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.5 --arch  bninception_frozen_normalize_double \
--embed_dim 512 --loss antico --bs 90 --warmup 0 --samples_per_class 2 --loss_ac_w_align 0.0035 --antico_w 1  --antico_eps 0.5 --antico_type batch_proxy

CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --no_train_metrics --source_path ../DataSet \
--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.5 --arch  resnet50_frozen_normalize_double \
--embed_dim 512 --loss antico --bs 90 --warmup 0 --samples_per_class 2 --loss_ac_w_align 0.0035 --antico_w 1  --antico_eps 0.5 --antico_type batch_proxy
 #-------------------
 #   CARS196
 #-------------------
CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --no_train_metrics --source_path ../DataSet \
--dataset cars196 --n_epochs 300 --tau 100 --gamma 0.5 --arch  bninception_frozen_normalize_double \
--embed_dim 512 --loss antico --bs 90 --warmup 0 --samples_per_class 2 --loss_ac_w_align 0.0035 --antico_w 1  --antico_eps 0.5 --antico_type batch_proxy

CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --no_train_metrics --source_path ../DataSet \
--dataset cars196 --n_epochs 300 --tau 100 --gamma 0.5 --arch  resnet50_frozen_normalize_double \
--embed_dim 512 --loss antico --bs 90 --warmup 0 --samples_per_class 2 --loss_ac_w_align 0.0035 --antico_w 1  --antico_eps 0.5 --antico_type batch_proxy
 #-------------------
 #   SOP
 #-------------------
 CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --no_train_metrics --source_path  ../DataSet \
--dataset online_products  --n_epochs 400 --tau 190 250 --gamma 0.2 --arch bninception_frozen_normalize_double \
--embed_dim 512 --loss nir_mcr2 --bs 90 --warmup 1 --loss_nir_w_align 0.0035 --loss_nir_lrmulti 1 --antico_w 1  --antico_eps 0.5 --antico_type batch_proxy
 CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --no_train_metrics --source_path  ../DataSet \
--dataset online_products  --n_epochs 400 --tau 190 250 --gamma 0.2 --arch resnet50_frozen_normalize_double \
--embed_dim 512--loss nir_mcr2 --bs 90 --warmup 1 --loss_nir_w_align 0.0035 --loss_nir_lrmulti 1 --antico_w 1  --antico_eps 0.5 --antico_type batch_proxy
#
#CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
#--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  bninception_frozen_normalize_double \
#--embed_dim 64  --loss nir_mcr2 --bs 90 --warmup 10 --samples_per_class 2 --loss_nir_w_align 0.0035
#CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
#--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  bninception_frozen_normalize_double \
#--embed_dim 128  --loss nir_mcr2 --bs 90 --warmup 10 --samples_per_class 2 --loss_nir_w_align 0.0035
#CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
#--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  bninception_frozen_normalize_double \
#--embed_dim 512  --loss nir_mcr2 --bs 90 --warmup 10 --samples_per_class 2 --loss_nir_w_align 0.0035
#CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
#--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  bninception_frozen_normalize_double \
#--embed_dim 1024  --loss nir_mcr2 --bs 90 --warmup 10 --samples_per_class 2 --loss_nir_w_align 0.0035
#CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
#--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  bninception_frozen_normalize \
#--embed_dim 64  --loss nir_mcr2 --bs 90 --warmup 10 --samples_per_class 2 --loss_nir_w_align 0.0035
#CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
#--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  bninception_frozen_normalize \
#--embed_dim 128  --loss nir_mcr2 --bs 90 --warmup 10 --samples_per_class 2 --loss_nir_w_align 0.0035
#CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
#--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  bninception_frozen_normalize \
#--embed_dim 512  --loss nir_mcr2 --bs 90 --warmup 10 --samples_per_class 2 --loss_nir_w_align 0.0035
#CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
#--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  bninception_frozen_normalize \
#--embed_dim 1024  --loss nir_mcr2 --bs 90 --warmup 10 --samples_per_class 2 --loss_nir_w_align 0.0035
#CUDA_VISIBLE_DEVICES=1 python main.py --seed 7 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
#--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  resnet50_frozen_normalize_double \
#--embed_dim 512  --loss nir_mcr2 --bs 90 --warmup 15 --samples_per_class 2 --loss_nir_w_align 0.0035
#CUDA_VISIBLE_DEVICES=1 python main.py --seed 8 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
#--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  resnet50_frozen_normalize_double \
#--embed_dim 512  --loss nir_mcr2 --bs 90 --warmup 15 --samples_per_class 2 --loss_nir_w_align 0.0035
#CUDA_VISIBLE_DEVICES=1 python main.py --seed 9 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
#--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  resnet50_frozen_normalize_double \
#--embed_dim 512  --loss nir_mcr2 --bs 90 --warmup 15 --samples_per_class 2 --loss_nir_w_align 0.0035
#CUDA_VISIBLE_DEVICES=1 python main.py --seed 10 --no_train_metrics --source_path /home/jxr/proj/NIR/DataSet \
#--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch  resnet50_frozen_normalize_double \
#--embed_dim 512  --loss nir_mcr2 --bs 90 --warmup 15 --samples_per_class 2 --loss_nir_w_align 0.0035


# #-------------------
# #   CARS196
# #-------------------
# ### BNInception, dim = 128
# ## ProxyAnchor Baseline
# python main.py --seed 0 --no_train_metrics --source_path  /home/jxr/proj/NIR/DataSet \
# --dataset cars196 --n_epochs 100 --tau 100 --gamma 0.1 --arch  bninception_frozen_normalize \
# --embed_dim 120 --loss oproxy --bs 90
#
# ## ProxyAnchor + NIR
# python main.py --seed 0 --no_train_metrics --source_path  /home/jxr/proj/NIR/DataSet \
# --dataset cars196 --n_epochs 100 --tau 100 --gamma 0.1 --arch  bninception_frozen_normalize \
# --embed_dim 128 --loss nir --bs 90 --warmup 1 --loss_nir_w_align 0.01
#
# ## ProxyAnchor + NIR | DoublePool
# CUDA_VISIBLE_DEVICES=0 python main.py --seed 1 --no_train_metrics --source_path  /home/jxr/proj/NIR/DataSet \
# --dataset cars196 --n_epochs 350 --tau 150 --gamma 0.1 --arch  bninception_frozen_normalize \
# --embed_dim 512 --loss nir_mcr2 --bs 90 --warmup 50 --loss_nir_w_align 0.01
#
# ### ResNet50, dim = 512
# ## ProxyAnchor + NIR
# CUDA_VISIBLE_DEVICES=0 python main.py --seed 1 --no_train_metrics --source_path  /home/jxr/proj/NIR/DataSet \
# --dataset cars196 --n_epochs 300 --tau 100 --gamma 0.1 --arch  resnet50_frozen_normalize \
# --embed_dim 512 --loss nir --bs 90 --warmup 0 --loss_nir_w_align 0.01
#
# ## ProxyAnchor + NIR | DoublePool
# CUDA_VISIBLE_DEVICES=0 python main.py --seed 6 --no_train_metrics --source_path  /home/jxr/proj/NIR/DataSet \
# --dataset cars196 --n_epochs 600 --tau 150 --gamma 0.1 --arch  resnet50_frozen_normalize_double \
# --embed_dim 512 --loss nir_mcr2 --bs 250 --warmup 50 --loss_nir_w_align 0.0075
#
#
#
##-------------------
##   SOP
##-------------------
#### BNInception, dim = 128
### ProxyAnchor + NIR
#python main.py --seed 0 --no_train_metrics --source_path  /home/jxr/proj/NIR/DataSet \
#--dataset online_products  --n_epochs 300 --tau 140 230 --gamma 0.2 --arch  bninception_frozen_normalize \
#--embed_dim 128 --loss nir --bs 90 --warmup 5 --loss_nir_w_align 0.3 --loss_nir_lrmulti 1
#
### ProxyAnchor + NIR | DoublePool
#python main.py --seed 0 --no_train_metrics --source_path  /home/jxr/proj/NIR/DataSet \
#--dataset online_products  --n_epochs 300 --tau 140 230 --gamma 0.2 --arch  bninception_frozen_normalize_double \
#--embed_dim 128 --loss nir --bs 90 --warmup 5 --loss_nir_w_align 0.3 --loss_nir_lrmulti 1
#
#### BNInception, dim = 512
### ProxyAnchor + NIR
#python main.py --seed 0 --no_train_metrics --source_path  /home/jxr/proj/NIR/DataSet \
#--dataset online_products  --n_epochs 300 --tau 140 230 --gamma 0.2 --arch  bninception_frozen_normalize \
#--embed_dim 512 --loss nir --bs 90 --warmup 5 --loss_nir_w_align 0.3 --loss_nir_lrmulti 1
#
### ProxyAnchor + NIR | DoublePool
# CUDA_VISIBLE_DEVICES=2 python main.py --seed 5 --no_train_metrics --source_path  /home/jxr/proj/NIR/DataSet \
#--dataset online_products  --n_epochs 350 --tau 190 280 --gamma 0.2 --arch  resnet50_frozen_normalize_double \
#--embed_dim 512 --loss nir_mcr2 --bs 400 --warmup 50 --loss_nir_w_align 0.3 --loss_nir_lrmulti 1
# CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path  /home/jxr/proj/NIR/DataSet \
#--dataset online_products  --n_epochs 350 --tau 150 240 --gamma 0.2 --arch resnet50_frozen_normalize \
#--embed_dim 512 --loss nir_mcr2 --bs 90 --warmup 10 --loss_nir_w_align 0.3 --loss_nir_lrmulti 1
# CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path  /home/jxr/proj/NIR/DataSet \
#--dataset online_products  --n_epochs 350 --tau 160 250 --gamma 0.2 --arch resnet50_frozen_normalize \
#--embed_dim 512 --loss nir_mcr2 --bs 90 --warmup 20 --loss_nir_w_align 0.3 --loss_nir_lrmulti 1
# CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path  /home/jxr/proj/NIR/DataSet \
#--dataset online_products  --n_epochs 400 --tau 190 250 --gamma 0.2 --arch bninception_frozen_normalize_double \
#--embed_dim 64 --loss nir_mcr2 --bs 90 --warmup 30 --loss_nir_w_align 0.3 --loss_nir_lrmulti 1
# CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path  /home/jxr/proj/NIR/DataSet \
#--dataset online_products  --n_epochs 400 --tau 190 250 --gamma 0.2 --arch bninception_frozen_normalize_double \
#--embed_dim 128 --loss nir_mcr2 --bs 90 --warmup 30 --loss_nir_w_align 0.3 --loss_nir_lrmulti 1
# CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path  /home/jxr/proj/NIR/DataSet \
#--dataset online_products  --n_epochs 400 --tau 190 250 --gamma 0.2 --arch bninception_frozen_normalize_double \
#--embed_dim 512 --loss nir_mcr2 --bs 90 --warmup 30 --loss_nir_w_align 0.3 --loss_nir_lrmulti 1
# CUDA_VISIBLE_DEVICES=1 python main.py --seed 0 --no_train_metrics --source_path  /home/jxr/proj/NIR/DataSet \
#--dataset online_products  --n_epochs 400 --tau 190 250 --gamma 0.2 --arch bninception_frozen_normalize_double \
#--embed_dim 1024 --loss nir_mcr2 --bs 90 --warmup 30 --loss_nir_w_align 0.3 --loss_nir_lrmulti 1
