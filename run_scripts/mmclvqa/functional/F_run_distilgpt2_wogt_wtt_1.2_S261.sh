ROOT=/Users/stan
DEVICE=0
if [ ! -f "$ROOT/exp/clvqa/save/functional/setting_261_roslak/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.2/unicl_object/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/functional/cl_object_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=roslak \
 training.CL.restore_rate=1.2 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token/distilgpt2_replay/distilgpt2_functional_roslak \
 training.CL.restore_paths=roslak_REPLAY[r]_AT[o].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/stand_alone/functional/unicl_relation/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/functional/setting_261_roslak/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.2/unicl_object \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/functional/setting_261_roslak/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.2/unicl_scenetext/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/functional/cl_scenetext_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=roslak \
 training.CL.restore_rate=1.2 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token/distilgpt2_replay/distilgpt2_functional_roslak \
 training.CL.restore_paths=roslak_REPLAY[r]_AT[s].npy,roslak_REPLAY[o]_AT[s].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/functional/setting_261_roslak/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.2/unicl_object/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/functional/setting_261_roslak/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.2/unicl_scenetext \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/functional/setting_261_roslak/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.2/unicl_logical/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/functional/cl_logical_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=roslak \
 training.CL.restore_rate=1.2 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token/distilgpt2_replay/distilgpt2_functional_roslak \
 training.CL.restore_paths=roslak_REPLAY[r]_AT[l].npy,roslak_REPLAY[o]_AT[l].npy,roslak_REPLAY[s]_AT[l].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/functional/setting_261_roslak/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.2/unicl_scenetext/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/functional/setting_261_roslak/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.2/unicl_logical \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/functional/setting_261_roslak/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.2/unicl_attribute/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/functional/cl_attribute_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=roslak \
 training.CL.restore_rate=1.2 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token/distilgpt2_replay/distilgpt2_functional_roslak \
 training.CL.restore_paths=roslak_REPLAY[r]_AT[a].npy,roslak_REPLAY[o]_AT[a].npy,roslak_REPLAY[s]_AT[a].npy,roslak_REPLAY[l]_AT[a].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/functional/setting_261_roslak/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.2/unicl_logical/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/functional/setting_261_roslak/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.2/unicl_attribute \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/functional/setting_261_roslak/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.2/unicl_knowledge/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/functional/cl_knowledge_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=roslak \
 training.CL.restore_rate=1.2 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token/distilgpt2_replay/distilgpt2_functional_roslak \
 training.CL.restore_paths=roslak_REPLAY[r]_AT[k].npy,roslak_REPLAY[o]_AT[k].npy,roslak_REPLAY[s]_AT[k].npy,roslak_REPLAY[l]_AT[k].npy,roslak_REPLAY[a]_AT[k].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/functional/setting_261_roslak/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.2/unicl_attribute/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/functional/setting_261_roslak/distilgpt2_replay_qag_seq_not_use_gt_task_token_1.2/unicl_knowledge \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



python -c 'from eval_os import *; stage_sweep(cl_setting="functional", setting_idx=261, abbr_seq="roslak", device="'${DEVICE}'", model_name="unicl", save_dir="'${ROOT}'/exp/clvqa", val_exp="distilgpt2_replay_qag_seq_not_use_gt_task_token_1.2", test_stand_alone=False, test_reg=False)' 
