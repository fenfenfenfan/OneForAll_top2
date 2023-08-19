export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
config=configs/train_convxl_m2f_e60_revise_dino256_pre_aug1_v2_1280_mosaic_noseg.py
python3 -m paddle.distributed.launch --log_dir=./logs/vitbase_jointraining_mosaic --gpus="0,1,2,3,4,5,6,7" tools/ufo_train.py --config-file ${config}