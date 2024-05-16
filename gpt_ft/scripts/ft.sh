ROOT=/mnt/lustre/tangyang2/hjq/embedding/gpt_ft
export PYTHONPATH=$ROOT:${PYTHONPATH}

# SLURM Parameter
GPUS_PER_NODE=8
if [ -z "$WORLD_SIZE" ]; then  
    WORLD_SIZE=1  
    RANK=0
    MASTER_ADDR=127.0.0.1
    MASTER_PORT=6000
fi

# WorkSpace Param
JOBNAME='Fine-Tuning-Gritlm'
OUTPUTDIR=$ROOT/outputs
LOGDIR=$ROOT/logs
mkdir -p $LOGDIR

# Hyper Parameter Start
PRETRAIN_MODEL_NAME=GritLM-7B
EPOCHS=1
BATCH_SIZE=40
LR=1e-5
NEG_NUM=2
DS_PATH=$ROOT/ds_config_zero1.json
TEMPRATURE=0.01
OUTPUT_NAME=$PRETRAIN_MODEL_NAME.e$EPOCHS.lr$LR.B$BATCH_SIZE.Neg$NEG_NUM.G$WORLD_SIZE.$JOBNAME
MAX_LENGTH=512
META_PATHS=(
meta_lists/piccolo-ft.txt
)
ROOT_DIRS=(
data_example/
)
# Hyper Parameter End 


# Model Parameter
model_args=(
    "--model_name_or_path" "/mnt/lustre/tangyang2/hjq/model/$PRETRAIN_MODEL_NAME/"
    "--loss_type=hardneg_softmax"
    "--temperature=$TEMPRATURE"
    "--max_length=$MAX_LENGTH"
    "--query_prefix=''"
    "--doc_prefix=''"
    "--use_scaling_layer=False"
    "--use_mrl=True"
)

# Data Parameter
data_args=(
    "--meta_paths" "${META_PATHS[@]}"
    "--root_dirs" "${ROOT_DIRS[@]}"
    "--neg_num=$NEG_NUM"
    "--use_all_pair=True"
)

# Train Parameter
train_args=(
    "--bf16"
    "--gradient_checkpointing=True"
    "--with_instruction=True"
    "--output_dir=$OUTPUTDIR/$OUTPUT_NAME"
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
    "--use_optimum=False"
)

# Unified Parameters
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
    $ROOT/uniem/train_ste.py \
    "

echo $CMD

bash -c "$LAUNCHER $CMD ${all_args[*]}" 2>&1 | tee -a $LOGDIR/$OUTPUT_NAME.txt