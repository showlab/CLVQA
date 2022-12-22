# --- seq distil gpt2, use sampled sg, REPORT in paper
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting scene --task_seq beacfd --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/not_use_gt/QAG_scene_task_token   --add_task_tokens --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting scene --task_seq beadcf --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/not_use_gt/QAG_scene_task_token   --add_task_tokens --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting scene --task_seq bedfca --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/not_use_gt/QAG_scene_task_token   --add_task_tokens --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting scene --task_seq ecdfab --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/not_use_gt/QAG_scene_task_token   --add_task_tokens --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting scene --task_seq bdfcae --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/not_use_gt/QAG_scene_task_token   --add_task_tokens --n_train_epochs 15


CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting functional --task_seq oarlks --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token   --add_task_tokens --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting functional --task_seq roslak --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token   --add_task_tokens --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting functional --task_seq rklsao --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token   --add_task_tokens --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting functional --task_seq rsolak --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token   --add_task_tokens --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting functional --task_seq lkosra --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token   --add_task_tokens --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting functional --task_seq kaorls --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token   --add_task_tokens --n_train_epochs 15


# --- seq distil gpt2, use gt sg
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting scene --task_seq abcdef --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/use_gt/QAG_scene_task_token   --add_task_tokens --use_gt --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting scene --task_seq beacfd --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/use_gt/QAG_scene_task_token   --add_task_tokens --use_gt --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting scene --task_seq beadcf --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/use_gt/QAG_scene_task_token   --add_task_tokens --use_gt --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting scene --task_seq bedfca --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/use_gt/QAG_scene_task_token   --add_task_tokens --use_gt --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting scene --task_seq ecdfab --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/use_gt/QAG_scene_task_token   --add_task_tokens --use_gt --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting scene --task_seq bdfcae --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/use_gt/QAG_scene_task_token   --add_task_tokens --use_gt --n_train_epochs 15


CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting functional --task_seq oarlks --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/use_gt/QAG_functional_task_token   --add_task_tokens --use_gt --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting functional --task_seq roslak --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/use_gt/QAG_functional_task_token   --add_task_tokens --use_gt --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting functional --task_seq rklsao --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/use_gt/QAG_functional_task_token   --add_task_tokens --use_gt --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting functional --task_seq rsolak --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/use_gt/QAG_functional_task_token   --add_task_tokens --use_gt --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting functional --task_seq lkosra --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/use_gt/QAG_functional_task_token   --add_task_tokens --use_gt --n_train_epochs 15
CUDA_VISIBLE_DEVICES=0 python train.py --cl_setting functional --task_seq kaorls --model_name distilgpt2 --model_dir_root  /Users/stan/exp/clvqa/QAG_seq/use_gt/QAG_functional_task_token   --add_task_tokens --use_gt --n_train_epochs 15





