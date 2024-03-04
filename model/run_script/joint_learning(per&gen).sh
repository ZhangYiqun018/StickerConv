CUDA_VISIBLE_DEVICES=3 \
    python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port='29501' \
        train.py \
            --config pegs/configs/train/joint_learning(per&gen).yaml