# Benchmarking Run Parameters

### Low Complexity Example:
    python create_bench.py --train --stim_dir='data/shapenet_handpicked_val' --tasks_dir='benchmarking/temp/low_tasks_all' --trials_dir='benchmarking/temp/low_trials_all' --config_path='benchmarking/configs/low_complexity.json' --max_memory=5 --max_len=6 --n_trials=1000 --n_tasks=1000 --max_depth=5 --max_op=15 --max_switch=0 --switch_threshold=1.0 --select_limit --features='all' --min_bool_ops=1 --max_bool_ops=1 --force_balance --non_bool_actions

### Medium Complexity Example:
    python create_bench.py --train --stim_dir='data/shapenet_handpicked_val' --tasks_dir='benchmarking/temp/medium_tasks_all' --trials_dir='benchmarking/temp/medium_trials_all' --config_path='benchmarking/configs/medium_complexity.json' --max_memory=7 --max_len=8 --n_trials=1000 --n_tasks=1000 --max_depth=7 --max_op=15 --max_switch=1 --switch_threshold=1.0 --select_limit --features='all' --min_bool_ops=1 --max_bool_ops=1 --force_balance --non_bool_actions

### Medium Complexity Example:
    python create_bench.py --train --stim_dir='data/shapenet_handpicked_val' --tasks_dir='benchmarking/temp/high_tasks_all' --trials_dir='benchmarking/temp/high_trials_all' --config_path='benchmarking/configs/high_complexity.json' --max_memory=8 --max_len=9 --n_trials=1000 --n_tasks=1000 --max_depth=8 --max_op=15 --max_switch=1 --switch_threshold=1.0 --select_limit --features='all' --min_bool_ops=1 --max_bool_ops=2 --force_balance