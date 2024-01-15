cp -r /datas/zyq/research/chat_meme/evaluates/generate_response_zyq.py /datas/llm_datasets/kfhcode/memechat/generate_response_zyq.py
CUDA_VISIBLE_DEVICES=6 python /datas/llm_datasets/kfhcode/memechat/generate_response_zyq.py \
    --data_path /datas/zyq/research/chat_meme/evaluates/temp/tran_converted_test.json > nohup2.out 2>&1 &