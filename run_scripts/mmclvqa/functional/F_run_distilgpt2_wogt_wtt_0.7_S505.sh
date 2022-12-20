ROOT=/Users/stan
DEVICE=0
if [ ! -f "$ROOT/exp/clvqa/save/functional/setting_505_kaorls/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_attribute/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/functional/cl_attribute_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=kaorls \
 training.CL.restore_rate=0.7 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token/distilgpt2_replay/distilgpt2_functional_kaorls \
 training.CL.restore_paths=kaorls_REPLAY[k]_AT[a].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/stand_alone/functional/unicl_knowledge/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/functional/setting_505_kaorls/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_attribute \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/functional/setting_505_kaorls/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_object/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/functional/cl_object_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=kaorls \
 training.CL.restore_rate=0.7 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token/distilgpt2_replay/distilgpt2_functional_kaorls \
 training.CL.restore_paths=kaorls_REPLAY[k]_AT[o].npy,kaorls_REPLAY[a]_AT[o].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/functional/setting_505_kaorls/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_attribute/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/functional/setting_505_kaorls/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_object \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/functional/setting_505_kaorls/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_relation/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/functional/cl_relation_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=kaorls \
 training.CL.restore_rate=0.7 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token/distilgpt2_replay/distilgpt2_functional_kaorls \
 training.CL.restore_paths=kaorls_REPLAY[k]_AT[r].npy,kaorls_REPLAY[a]_AT[r].npy,kaorls_REPLAY[o]_AT[r].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/functional/setting_505_kaorls/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_object/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/functional/setting_505_kaorls/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_relation \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/functional/setting_505_kaorls/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_logical/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/functional/cl_logical_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=kaorls \
 training.CL.restore_rate=0.7 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token/distilgpt2_replay/distilgpt2_functional_kaorls \
 training.CL.restore_paths=kaorls_REPLAY[k]_AT[l].npy,kaorls_REPLAY[a]_AT[l].npy,kaorls_REPLAY[o]_AT[l].npy,kaorls_REPLAY[r]_AT[l].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/functional/setting_505_kaorls/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_relation/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/functional/setting_505_kaorls/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_logical \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



if [ ! -f "$ROOT/exp/clvqa/save/functional/setting_505_kaorls/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_scenetext/unicl_final.pth" ] ; then 
 CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config=EXP_CONFIG/functional/cl_scenetext_unicl_standalone.yaml \
 model=unicl \
 dataset=clvqa \
 training.CL.use_cl=True \
 training.CL.use_callback=False \
 training.CL.use_replay=True \
 training.CL.replay_method=restore_with_prob \
 training.CL.task_order=kaorls \
 training.CL.restore_rate=0.7 \
 training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/not_use_gt/QAG_functional_task_token/distilgpt2_replay/distilgpt2_functional_kaorls \
 training.CL.restore_paths=kaorls_REPLAY[k]_AT[s].npy,kaorls_REPLAY[a]_AT[s].npy,kaorls_REPLAY[o]_AT[s].npy,kaorls_REPLAY[r]_AT[s].npy,kaorls_REPLAY[l]_AT[s].npy \
 dataset_config.clvqa.use_mask_img=True \
 dataset_config.clvqa.mask_img_prob=0.15 \
 run_type=train_val \
 checkpoint.resume_file=$ROOT/exp/clvqa/save/functional/setting_505_kaorls/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_logical/unicl_final.pth \
 env.save_dir=$ROOT/exp/clvqa/save/functional/setting_505_kaorls/distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7/unicl_scenetext \
 training.checkpoint_interval=4000 \
 training.callbacks=[] 
fi 



python -c 'from eval_os import *; stage_sweep(cl_setting="functional", setting_idx=505, abbr_seq="kaorls", device="'${DEVICE}'", model_name="unicl", save_dir="'${ROOT}'/exp/clvqa", val_exp="distilgpt2_replay_qag_seq_not_use_gt_task_token_0.7", test_stand_alone=False, test_reg=False)' 
