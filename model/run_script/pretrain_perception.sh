CUDA_VISIBLE_DEVICES=0,1 \
    python -m torch.distributed.run \
        --nproc_per_node 2 \
        --master_port='29501' \
        train.py \
            --config pegs/configs/train/pretrain_perception.yaml
