export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
config=configs/test_convnext_xl_mask2f_1280_tta.py
python3 -m paddle.distributed.launch --log_dir=./logs/vitbase_jointraining_tta --gpus="0,1,2,3,4,5,6,7" tools/ufo_test.py --config-file ${config}