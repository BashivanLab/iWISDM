# Benchmarking Run Parameters

### Single Frame Example
    python create_bench.py --stim_dir='../data/shapenet_handpicked' --tasks_dir='./tasks/single_cat' --trials_dir='temp/single_cat' --config_path='configs/single_frame_cat.json' --min_len=1 --max_len=1 --n_trials=100 --n_tasks=10 --features='cat' --min_joint_ops=0 --max_joint_ops=0

### Low Complexity Example:
    python create_bench.py --stim_dir='../data/shapenet_handpicked' --tasks_dir='./tasks/low_all' --trials_dir='temp/low_all' 
    --config_path='configs/low_complexity_all.json' --min_len=6 --max_len=6  --n_trials=100 --n_tasks=100 --features='all' --min_joint_ops=1 --max_joint_ops=1

### Medium Complexity Example:
    python create_bench.py --stim_dir='../data/shapenet_handpicked' --tasks_dir='./tasks/medium_all' --trials_dir='temp/medium_all' 
    --config_path='configs/medium_complexity_all.json' --min_len=8 --max_len=8 --n_trials=100 --n_tasks=100 --features='all' --min_joint_ops=1 --max_joint_ops=1

### High Complexity Example:
    python create_bench.py --stim_dir='../data/shapenet_handpicked' --tasks_dir='./tasks/high_all' --trials_dir='temp/high_all' 
    --config_path='configs/high_complexity_all.json' --min_len=9 --max_len=9 --n_trials=100 --n_tasks=100 --features='all' --min_joint_ops=1 --max_joint_ops=2 --non_bool_actions