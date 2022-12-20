ROOT=/Users/stan
DEVICE=0
if [ ! -f "$ROOT/exp/clvqa/save/scene/setting_1_abcdef/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.9/unicl_b#Workplace/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/scene/cl_b#Workplace_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=abcdef \
 training.CL.restore_rate=0.9 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_scene_task_token/distilgpt2_replay/distilgpt2_scene_abcdef \
 training.CL.restore_paths=abcdef_REPLAY[a]_AT[b].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/stand_alone/scene/unicl_a#ShopAndDining/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/scene/setting_1_abcdef/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.9/unicl_b#Workplace \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/scene/setting_1_abcdef/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.9/unicl_c#HomeOrHotel/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/scene/cl_c#HomeOrHotel_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=abcdef \
 training.CL.restore_rate=0.9 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_scene_task_token/distilgpt2_replay/distilgpt2_scene_abcdef \
 training.CL.restore_paths=abcdef_REPLAY[a]_AT[c].npy,abcdef_REPLAY[b]_AT[c].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/scene/setting_1_abcdef/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.9/unicl_b#Workplace/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/scene/setting_1_abcdef/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.9/unicl_c#HomeOrHotel \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/scene/setting_1_abcdef/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.9/unicl_d#Transportation/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/scene/cl_d#Transportation_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=abcdef \
 training.CL.restore_rate=0.9 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_scene_task_token/distilgpt2_replay/distilgpt2_scene_abcdef \
 training.CL.restore_paths=abcdef_REPLAY[a]_AT[d].npy,abcdef_REPLAY[b]_AT[d].npy,abcdef_REPLAY[c]_AT[d].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/scene/setting_1_abcdef/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.9/unicl_c#HomeOrHotel/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/scene/setting_1_abcdef/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.9/unicl_d#Transportation \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/scene/setting_1_abcdef/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.9/unicl_e#SportAndLeisure/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/scene/cl_e#SportAndLeisure_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=abcdef \
 training.CL.restore_rate=0.9 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_scene_task_token/distilgpt2_replay/distilgpt2_scene_abcdef \
 training.CL.restore_paths=abcdef_REPLAY[a]_AT[e].npy,abcdef_REPLAY[b]_AT[e].npy,abcdef_REPLAY[c]_AT[e].npy,abcdef_REPLAY[d]_AT[e].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/scene/setting_1_abcdef/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.9/unicl_d#Transportation/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/scene/setting_1_abcdef/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.9/unicl_e#SportAndLeisure \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/scene/setting_1_abcdef/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.9/unicl_f#Outdoors/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/scene/cl_f#Outdoors_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=abcdef \
 training.CL.restore_rate=0.9 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_scene_task_token/distilgpt2_replay/distilgpt2_scene_abcdef \
 training.CL.restore_paths=abcdef_REPLAY[a]_AT[f].npy,abcdef_REPLAY[b]_AT[f].npy,abcdef_REPLAY[c]_AT[f].npy,abcdef_REPLAY[d]_AT[f].npy,abcdef_REPLAY[e]_AT[f].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/scene/setting_1_abcdef/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.9/unicl_e#SportAndLeisure/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/scene/setting_1_abcdef/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.9/unicl_f#Outdoors \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



python -c 'from eval_os import *; stage_sweep(cl_setting="scene", setting_idx=1, abbr_seq="abcdef", device="'${DEVICE}'", model_name="unicl", save_dir="'${ROOT}'/exp/clvqa", val_exp="distilgpt2_replay_qag_seq_not_use_gt_task_token_0.9", test_stand_alone=False, test_reg=False)' 
