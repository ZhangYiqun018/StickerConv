# cp -r /datas/zyq/research/chat_meme/evaluates/generate_response_zyq.py /datas/llm_datasets/kfhcode/memechat/generate_response_zyq.py
RUNNAME=1\_31\_19
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=2 

python /datas/llm_datasets/memechat/codes/generate_response.py \
    --data_path /datas/zyq/research/chat_meme/evaluates/util/pegs_1.5_test_v3.json \
    --output_path /datas/zyq/research/chat_meme/evaluates/outputs/baseline/pegs_retrieve_$RUNNAME.json > nohup\_$RUNNAME.out 2>&1 &