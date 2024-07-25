USE_TF=0
cache_dir=$1
model_name=$2
output_dir=$3

deepspeed --master_port 29510 \
		./ds_train.py \
		--cache_dir $cache_dir \
        --model_name_or_path $model_name \
        --output_dir $output_dir \
        --do_train \
		--do_eval \
		--save_total_limit=100 \
        --train_file ../data/fast_system.train.jsonl \
		--validation_file ../data/fast_system.test.jsonl \
		--predict_with_generate 0 \
        --learning_rate 1e-4 \
		--adam_eps 1e-06 \
        --overwrite_output_dir \
        --max_source_length 1024 \
        --max_target_length 30 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 32 \
	--metric_for_best_model eval_loss \
	--greater_is_better=False \
	--deepspeed zero_2_bf16.json \
	--gradient_accumulation_steps 4 \
	--num_train_epochs 30 \
	--logging_steps 1 \
	--load_best_model_at_end=True \
	--save_strategy=steps \
	--evaluation_strategy=steps \
	--save_steps 100 \
	--eval_steps 100 \
	--seed 42 \
	--report_to wandb \
	--run_name $model_name
