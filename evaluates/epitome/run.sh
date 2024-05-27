TYPE=pegs_v1.5_ret
CUDA_VISIBLE_DEVICES=4 python evaluates.py \
    --datatype $TYPE \
    --data_dir /datas/zyq/research/chat_meme/evaluates/outputs/baseline/pegs_v1.5_ret_full_anno.json \
    --batch_size 4 > nohup_epitome\_$TYPE.out 2>&1 &