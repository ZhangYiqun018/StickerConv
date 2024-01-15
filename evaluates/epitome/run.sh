CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --datatype pegs_p \
    --data_dir /datas/zyq/research/chat_meme/prediction_anno_full.json \
    --batch_size 4 > nohup_epitome_pegs_p.out 2>&1 &