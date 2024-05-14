ROOT=/mnt/lustre/huangjunqin/piccolo
export PYTHONPATH=$ROOT:${PYTHONPATH}

# SLURM Parameter
GPUS_PER_NODE=1
if [ -z "$WORLD_SIZE" ]; then  
    WORLD_SIZE=1  
    RANK=0
    MASTER_ADDR=127.0.0.1
    MASTER_PORT=6000
fi

# Hyper Parameter Start
MODEL_NAME_OR_PATH=/mnt/lustre/huangjunqin/test/piccolo2-large-zh-0417
EPOCHS=3
BATCH_SIZE=8
LR=1e-5
NEG_NUM=1
DS_PATH=$ROOT/ds_config_zero1.json
MAX_LENGTH=512
META_PATHS=(
meta_lists/piccolo-ft.txt
)

ROOT_DIRS=(
data_example/
)
# Hyper Parameter End 


model_args=(
    "--model_name_or_path" $MODEL_NAME_OR_PATH
    "--max_length=$MAX_LENGTH"
    "--query_prefix=''"
    "--doc_prefix=''"
    "--use_scaling_layer=True"
    "--use_mrl=True"
)

data_args=(
    "--meta_paths" "${META_PATHS[@]}"
    "--root_dirs" "${ROOT_DIRS[@]}"
    "--neg_num=$NEG_NUM"
)

train_args=(
    "--fp16"
    "--gradient_checkpointing=True"
    "--output_dir=output_test"
    "--num_train_epochs=$EPOCHS"
    "--dataloader_num_workers=0"
    "--batch_size=$BATCH_SIZE"
    "--learning_rate=$LR"
    "--deepspeed=$DS_PATH"
    "--logging_steps=500"
    "--save_safetensors=False"
    "--report_to=tensorboard"
    "--save_strategy=epoch"
    "--per_device_train_batch_size=1"
)

all_args=("${model_args[@]}" "${data_args[@]}" "${train_args[@]}")


export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $WORLD_SIZE \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

export CMD=" \
    $ROOT/finetune/train.py \
    "

echo $CMD

bash -c "$LAUNCHER $CMD ${all_args[*]}"