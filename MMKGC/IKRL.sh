DATA_DIR=dataset

MODEL_NAME=$4
DATASET_NAME=$6
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KGELitModel
MAX_EPOCHS=10000
EMB_DIM=500
IMG_DIM=$3
IMG=$2
REL=$5
LOSS=Adv_Loss
ADV_TEMP=1.0
TRAIN_BS=1024
EVAL_BS=128
NUM_NEG=32
MARGIN=6.0
LR=2e-5
CHECK_PER_EPOCH=20
NUM_WORKERS=16
DATA_CHOICE=$7
GPU=$1


CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --img_dim $IMG_DIM \
    --IMG $IMG \
    --rel_number $REL \
    --loss $LOSS \
    --adv_temp $ADV_TEMP \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --num_neg $NUM_NEG \
    --margin $MARGIN \
    --lr $LR \
    --check_per_epoch $CHECK_PER_EPOCH \
    --num_workers $NUM_WORKERS \
    --data_choice $DATA_CHOICE \
    --save_config \