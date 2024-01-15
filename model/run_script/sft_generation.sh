CUDA_VISIBLE_DEVICES=0 \
    python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port='29501' \
        train.py \
            --config pegs/configs/train/sft_generation.yaml