ROOT=/Users/stan
DEVICE=0
if [ ! -f "$ROOT/exp/clvqa/save/functional/setting_438_lkosra/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.0/unicl_knowledge/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/functional/cl_knowledge_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=lkosra \
 training.CL.restore_rate=1.0 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token/distilgpt2_replay/distilgpt2_functional_lkosra \
 training.CL.restore_paths=lkosra_REPLAY[l]_AT[k].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/stand_alone/functional/unicl_logical/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/functional/setting_438_lkosra/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.0/unicl_knowledge \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/functional/setting_438_lkosra/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.0/unicl_object/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/functional/cl_object_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=lkosra \
 training.CL.restore_rate=1.0 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token/distilgpt2_replay/distilgpt2_functional_lkosra \
 training.CL.restore_paths=lkosra_REPLAY[l]_AT[o].npy,lkosra_REPLAY[k]_AT[o].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/functional/setting_438_lkosra/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.0/unicl_knowledge/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/functional/setting_438_lkosra/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.0/unicl_object \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/functional/setting_438_lkosra/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.0/unicl_scenetext/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/functional/cl_scenetext_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=lkosra \
 training.CL.restore_rate=1.0 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token/distilgpt2_replay/distilgpt2_functional_lkosra \
 training.CL.restore_paths=lkosra_REPLAY[l]_AT[s].npy,lkosra_REPLAY[k]_AT[s].npy,lkosra_REPLAY[o]_AT[s].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/functional/setting_438_lkosra/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.0/unicl_object/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/functional/setting_438_lkosra/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.0/unicl_scenetext \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/functional/setting_438_lkosra/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.0/unicl_relation/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/functional/cl_relation_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=lkosra \
 training.CL.restore_rate=1.0 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token/distilgpt2_replay/distilgpt2_functional_lkosra \
 training.CL.restore_paths=lkosra_REPLAY[l]_AT[r].npy,lkosra_REPLAY[k]_AT[r].npy,lkosra_REPLAY[o]_AT[r].npy,lkosra_REPLAY[s]_AT[r].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/functional/setting_438_lkosra/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.0/unicl_scenetext/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/functional/setting_438_lkosra/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.0/unicl_relation \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/functional/setting_438_lkosra/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.0/unicl_attribute/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/functional/cl_attribute_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=lkosra \
 training.CL.restore_rate=1.0 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token/distilgpt2_replay/distilgpt2_functional_lkosra \
 training.CL.restore_paths=lkosra_REPLAY[l]_AT[a].npy,lkosra_REPLAY[k]_AT[a].npy,lkosra_REPLAY[o]_AT[a].npy,lkosra_REPLAY[s]_AT[a].npy,lkosra_REPLAY[r]_AT[a].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/functional/setting_438_lkosra/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.0/unicl_relation/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/functional/setting_438_lkosra/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.0/unicl_attribute \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



python -c 'from eval_os import *; stage_sweep(cl_setting="functional", setting_idx=438, abbr_seq="lkosra", device="'${DEVICE}'", model_name="unicl", save_dir="'${ROOT}'/exp/clvqa", val_exp="distilgpt2_replay_qag_seq_not_use_gt_task_token_1.0", test_stand_alone=False, test_reg=False)' 
