STARTIDX=41
ENDIDX=81
MODE=train
SEED=9099
nohup python main.py \
    --llm_config config_private.ini \
    --mode $MODE \
    --start_idx $STARTIDX \
    --end_idx $ENDIDX \
    --max_turn 6 \
    --llm_type azure \
    --shuffle \
    --seed $SEED \
    --profile_path ../dataset/profile/train_profile.json \
    --candidate_number 10 > nohup_$MODE\_$STARTIDX\_$ENDIDX\_shuffle_$SEED.out 2>&1 &