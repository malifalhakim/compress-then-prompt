for opt in adamw; do
for lr in 0.001; do
for steps in 30000; do
CUDA_VISIBLE_DEVICES=$1 python soft_prompt_learning.py \
    --model_name_or_path $2 \
    --model $3 \
    --dataset $4 \
    --eval_every_steps 100 \
    --seqlen 1024 \
    --soft_token_num $5 \
    --prompt_lr ${lr} \
    --max_steps ${steps} \
    --optimizer ${opt} \
    --output_dir ./gptq/${opt}_lr${lr}_steps${steps}_token${soft_token_num}/${dataset} \
    --per_device_train_batch_size 2 2>&1 | tee ./logs/log_gptq_${opt}_lr${lr}_${dataset}_steps${steps}_token${soft_token_num}.txt \
    --root . 
done 
done
done

# | tee ./logs/log_${opt}_lr${lr}_${dataset}_steps${steps}.txt
# --per_device_eval_batch_size 1 \