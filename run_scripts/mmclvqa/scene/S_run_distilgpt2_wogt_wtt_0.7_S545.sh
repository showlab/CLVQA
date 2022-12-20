ROOT=/Users/stan
DEVICE=0
if [ ! -f "$ROOT/exp/clvqa/save/scene/setting_545_ecdfab/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_c#HomeOrHotel/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/scene/cl_c#HomeOrHotel_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=ecdfab \
 training.CL.restore_rate=0.7 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_scene_task_token/distilgpt2_replay/distilgpt2_scene_ecdfab \
 training.CL.restore_paths=ecdfab_REPLAY[e]_AT[c].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/stand_alone/scene/unicl_e#SportAndLeisure/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/scene/setting_545_ecdfab/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_c#HomeOrHotel \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/scene/setting_545_ecdfab/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_d#Transportation/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/scene/cl_d#Transportation_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=ecdfab \
 training.CL.restore_rate=0.7 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_scene_task_token/distilgpt2_replay/distilgpt2_scene_ecdfab \
 training.CL.restore_paths=ecdfab_REPLAY[e]_AT[d].npy,ecdfab_REPLAY[c]_AT[d].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/scene/setting_545_ecdfab/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_c#HomeOrHotel/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/scene/setting_545_ecdfab/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_d#Transportation \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/scene/setting_545_ecdfab/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_f#Outdoors/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/scene/cl_f#Outdoors_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=ecdfab \
 training.CL.restore_rate=0.7 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_scene_task_token/distilgpt2_replay/distilgpt2_scene_ecdfab \
 training.CL.restore_paths=ecdfab_REPLAY[e]_AT[f].npy,ecdfab_REPLAY[c]_AT[f].npy,ecdfab_REPLAY[d]_AT[f].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/scene/setting_545_ecdfab/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_d#Transportation/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/scene/setting_545_ecdfab/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_f#Outdoors \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/scene/setting_545_ecdfab/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_a#ShopAndDining/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/scene/cl_a#ShopAndDining_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=ecdfab \
 training.CL.restore_rate=0.7 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_scene_task_token/distilgpt2_replay/distilgpt2_scene_ecdfab \
 training.CL.restore_paths=ecdfab_REPLAY[e]_AT[a].npy,ecdfab_REPLAY[c]_AT[a].npy,ecdfab_REPLAY[d]_AT[a].npy,ecdfab_REPLAY[f]_AT[a].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/scene/setting_545_ecdfab/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_f#Outdoors/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/scene/setting_545_ecdfab/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_a#ShopAndDining \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/scene/setting_545_ecdfab/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_b#Workplace/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/scene/cl_b#Workplace_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=ecdfab \
 training.CL.restore_rate=0.7 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_scene_task_token/distilgpt2_replay/distilgpt2_scene_ecdfab \
 training.CL.restore_paths=ecdfab_REPLAY[e]_AT[b].npy,ecdfab_REPLAY[c]_AT[b].npy,ecdfab_REPLAY[d]_AT[b].npy,ecdfab_REPLAY[f]_AT[b].npy,ecdfab_REPLAY[a]_AT[b].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/scene/setting_545_ecdfab/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_a#ShopAndDining/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/scene/setting_545_ecdfab/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_b#Workplace \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



python -c 'from eval_os import *; stage_sweep(cl_setting="scene", setting_idx=545, abbr_seq="ecdfab", device="'${DEVICE}'", model_name="unicl", save_dir="'${ROOT}'/exp/clvqa", val_exp="distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7", test_stand_alone=False, test_reg=False)' 
