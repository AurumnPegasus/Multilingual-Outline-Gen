
# Run mt5
python train.py --train_path ./langdomgen/train.json --val_path ./langdomgen/val.json --test_path ./langdomgen/test.json --tokenizer google/mt5-base --model google/mt5-base --is_mt5 1 --exp_name mt5-run --save_dir ./ --num_epochs 20 --train_batch_size 16 --val_batch_size 16 --test_batch_size 16 --max_source_length 32 --max_target_length 128 --n_gpus 4 --strategy ddp --sanity_run no --prediction_path ./prediction_files/

# Run mbart
python train.py --train_path ./langdomgen/train.json --val_path ./langdomgen/val.json --test_path ./langdomgen/test.json --tokenizer facebook/mbart-large-50 --model facebook/mbart-large-50 --is_mt5 0 --exp_name mbart-run --save_dir ./ --num_epochs 20 --train_batch_size 16 --val_batch_size 16 --test_batch_size 16 --max_source_length 32 --max_target_length 128 --n_gpus 4 --strategy ddp --sanity_run no --prediction_path ./prediction_files/
