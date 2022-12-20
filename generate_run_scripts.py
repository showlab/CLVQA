import os
import sys
import itertools
import pickle as pkl
import numpy as np

from mmf.common.CL_constant import ABBR2TASK

def ABBR2TASKList(cl_setting, abbr_seq):
    abbr_mapping = ABBR2TASK[cl_setting]
    taskList = [abbr_mapping[abbr] for abbr in abbr_seq]
    return taskList

def gen_standalone(device, cl_setting, model, **kwargs):
    root_dir = kwargs.get("root_dir", "/Users/stan")
    print(f"ROOT={root_dir}")
    print(f"DEVICE={device}")
    tmpl = (
        "CUDA_VISIBLE_DEVICES=$DEVICE mmf_run dataset=clvqa  \\\n"
            " model=unicl \\\n"
            " config=EXP_CONFIG/{}/cl_{}_{}_standalone.yaml \\\n" 
            " env.save_dir=/Users/stan/exp/clvqa/save/stand_alone/{}/{}_{} \\\n"
            " training.checkpoint_interval=4000 \\\n"
            " training.batch_size=32 \\\n"
            " training.callbacks=[] "
    )
    stages = ABBR2TASKList(cl_setting, list(ABBR2TASK[cl_setting].keys()))
    for stage in stages:
        run_script = tmpl.format(cl_setting, stage, model, cl_setting, model, stage)
        print("\n\n{}".format(run_script))


def gen_ft_seq(device, cl_setting, arrvs, setting_idx=1, **kwargs):
    root_dir = kwargs.get("root_dir", "/Users/stan")
    print(f"ROOT={root_dir}")
    print(f"DEVICE={device}")
    tmpl = (
        "if [ ! -f \"{}\" ] ; then \n"
        " CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config={} \\\n"
        " model=unicl \\\n"
        " dataset=clvqa \\\n"
        " training.CL.use_cl=False \\\n"
        " run_type=train_val \\\n"
        " checkpoint.resume_file={} \\\n"
        " env.save_dir={} \\\n"
        " training.CL.use_cl=True \\\n"
        " training.CL.use_icarl=False \\\n"
        " training.CL.reg_type=\"\" \\\n"
        " training.CL.use_callback=False \\\n"
        " training.CL.use_specific_optim=False \\\n"
        " training.CL.cur_task={} \\\n"
        " training.CL.task_order={} \\\n"
        " training.checkpoint_interval=4000 \\\n"
        " training.batch_size=32 \\\n"
        " training.callbacks=[] \n"
        "fi \n"
    )
    
    stages = ABBR2TASKList(cl_setting=cl_setting, abbr_seq=arrvs)
    arrv_seq = arrvs
    for idx, stage in enumerate(stages):
        if idx == 0:
            continue
        resume_file = None
        if idx == 1:
            resume_file = "$ROOT/exp/clvqa/save/stand_alone/{}/unicl_{}/unicl_final.pth".format(cl_setting, stages[idx-1])
        else:
            resume_file = "$ROOT/exp/clvqa/save/{}/setting_{}_{}/ft/unicl_{}/unicl_final.pth".format(cl_setting, setting_idx, arrv_seq, stages[idx-1])
        
        config_pth = "EXP_CONFIG/{}/cl_{}_unicl_standalone.yaml".format(cl_setting, stage)
        final_model_pth = "$ROOT/exp/clvqa/save/{}/setting_{}_{}/ft/unicl_{}/unicl_final.pth".format(cl_setting, setting_idx, arrv_seq, stage)
        save_dir = "$ROOT/exp/clvqa/save/{}/setting_{}_{}/ft/unicl_{}".format(cl_setting, setting_idx, arrv_seq, stage)
        run_script = tmpl.format(
            final_model_pth, 
            config_pth,
            resume_file, 
            save_dir,
            stage,
            arrv_seq
        )
        print("\n\n {}".format(run_script))
    
    # eval
    export_f = None
    if kwargs.get("log", False):
        export_f = "> $ROOT/results/{}_run_{}_S{}.txt".format(cl_setting[0].upper(), "ft_seq", setting_idx)
    else:
        export_f = ""
    print("python -c \'from eval_os import *; stage_sweep(cl_setting=\"{}\", setting_idx={}, abbr_seq=\"{}\", device=\'${{DEVICE}}\', model_name=\"unicl\", save_dir=\"\'${{ROOT}}\'/exp/clvqa\", val_exp=\"ft\", test_stand_alone=False, test_reg=False)\' {}".format(cl_setting, setting_idx, arrv_seq, export_f))
        
def gen_random_replay_w_prob(device, cl_setting, abbr_seq, model, replay_prob, replay_mask_img=False, use_gt_sg=False, setting_idx=1, **kwargs):
    root_dir = kwargs.get("root_dir", "/Users/stan")
    print(f"ROOT={root_dir}")
    print(f"DEVICE={device}")
    tmpl = (
        "if [ ! -f \"{}\" ] ; then \n"
        " CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config={} \\\n"
        " model={} \\\n"
        " dataset=clvqa \\\n"
        " run_type=train_val \\\n"
        " checkpoint.resume_file={} \\\n"
        " env.save_dir={} \\\n"
        " training.checkpoint_interval=4000 \\\n"
        " training.CL.use_cl=True \\\n"
        " training.CL.use_icarl=False \\\n"
        " training.CL.reg_type=\"\" \\\n"
        " training.CL.use_callback=False \\\n"
        " training.CL.use_specific_optim=False \\\n"
        " training.CL.cur_task={} \\\n"
        " training.CL.task_order={} \\\n"
        " training.CL.use_replay=True \\\n"
        " training.CL.replay_mask_img={} \\\n"
        " training.CL.replay_method=random \\\n"
        " training.CL.replay_rate={} \\\n"
        " training.callbacks=[] \\\n"
        " dataset_config.clvqa.use_gt_sg={} \n"
        "fi "
    )
    str_mask_img = "True" if replay_mask_img else "False"
    str_rnd_replay = "maskimg_rnd_replay" if replay_mask_img else "rnd_replay"
    if use_gt_sg: str_rnd_replay = "gtsg_" + str_rnd_replay
    str_use_gt_sg = "True" if use_gt_sg else "False"
    stages = ABBR2TASKList(cl_setting=cl_setting, abbr_seq=abbr_seq)
    arrv_seq = abbr_seq
    for idx, stage in enumerate(stages):
        if idx==0:
            continue

        resume_file = None
        if idx==1:
            resume_file = "$ROOT/exp/clvqa/save/stand_alone/{}/{}_{}/{}_final.pth".format(cl_setting, model, stages[idx-1], model)
        else:
            resume_file = "$ROOT/exp/clvqa/save/{}/setting_{}_{}/{}_{}/{}_{}/{}_final.pth".format(cl_setting, setting_idx, arrv_seq, str_rnd_replay, replay_prob, model, stages[idx-1], model)
       
        save_dir = "$ROOT/exp/clvqa/save/{}/setting_{}_{}/{}_{}/{}_{}".format(cl_setting, setting_idx, arrv_seq, str_rnd_replay,replay_prob, model, stage)
        task_name = stage
        task_order = arrv_seq
        config_pth = "EXP_CONFIG/{}/cl_{}_{}_standalone.yaml".format(cl_setting, stage, model)
        final_model_pth = os.path.join(save_dir, "unicl_final.pth")
        run_script = tmpl.format(
            final_model_pth,
            config_pth, 
            model, 
            resume_file, 
            save_dir, 
            task_name, 
            task_order, 
            str_mask_img,
            replay_prob,
            str_use_gt_sg
        )

        print("{}\n\n".format(run_script))
        # eval:
    exp_name = '{}_{}'.format(str_rnd_replay, replay_prob)

    export_f = None
    if kwargs.get("log", False):
        export_f = "> $ROOT/results/{}_run_{}_S{}.txt".format(cl_setting[0].upper(), exp_name, setting_idx)
    else:
        export_f = ""
    print("python -c \'from eval_os import *; stage_sweep(cl_setting=\"{}\", setting_idx={}, abbr_seq=\"{}\", device=\'${{DEVICE}}\', model_name=\"unicl\", save_dir=\"\'${{ROOT}}\'/exp/clvqa\", val_exp=\"{}\", test_stand_alone=False, test_reg=False)\' {}".format(cl_setting, setting_idx, arrv_seq, exp_name, export_f))
    
def gen_kmeans_replay_w_prob(device, cl_setting, abbr_seq, model, replay_prob, replay_mask_img=False, setting_idx=1, **kwargs):
    root_dir = kwargs.get("root_dir", "/Users/stan")
    print(f"ROOT={root_dir}")
    print(f"DEVICE={device}")
    tmpl = ( # reomve training.callbacks=[], by default, config files contain kmeans callback
        "if [ ! -f \"{}\" ] ; then \n"
        " CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config={} \\\n"
        " model={} \\\n"
        " dataset=clvqa \\\n"
        " run_type=train_val \\\n"
        " checkpoint.resume_file={} \\\n"
        " env.save_dir={} \\\n"
        " training.checkpoint_interval=4000 \\\n"
        " training.CL.use_cl=True \\\n"
        " training.CL.use_icarl=False \\\n"
        " training.CL.reg_type=\"\" \\\n"
        " training.CL.use_callback=False \\\n"
        " training.CL.use_specific_optim=False \\\n"
        " training.CL.cur_task={} \\\n"
        " training.CL.task_order={} \\\n"
        " training.CL.use_replay=True \\\n"
        " training.CL.replay_dir={} \\\n"
        " training.CL.replay_mask_img={} \\\n"
        " training.CL.replay_method=kmeans \\\n"
        " training.CL.replay_rate={} \\\n"
        " model_config.unicl.use_cls=True \\\n"
        " training.batch_size=24 \n"
        "fi "
    )
    str_mask_img = "True" if replay_mask_img else "False"
    str_replay = "maskimg_kmeans_replay" if replay_mask_img else "kmeans_replay"
    stages = ABBR2TASKList(cl_setting=cl_setting, abbr_seq=abbr_seq)
    arrv_seq = abbr_seq

    for idx, stage in enumerate(stages):
        ##########################
        resume_file = None
        replay_dir = None
        if idx==0:
            resume_file = "None"
            replay_dir = "\"\""
        else:
            resume_file = "$ROOT/exp/clvqa/save/{}/setting_{}_{}/{}_{}/{}_{}/{}_final.pth".format(cl_setting, setting_idx, arrv_seq, str_replay, replay_prob, model, stages[idx-1], model)
            replay_dir = "$ROOT/exp/clvqa/save/{}/setting_{}_{}/{}_{}/{}_{}".format(cl_setting, setting_idx, arrv_seq, str_replay, replay_prob, model, stages[idx-1])
        
        config_pth = "EXP_CONFIG/{}/cl_{}_{}_standalone.yaml".format(cl_setting, stage, model)        
        save_dir = "$ROOT/exp/clvqa/save/{}/setting_{}_{}/{}_{}/{}_{}".format(cl_setting, setting_idx, arrv_seq, str_replay, replay_prob, model, stage)
        task_name = stage
        task_order = arrv_seq
        final_model_pth = os.path.join(save_dir, "unicl_final.pth")
        run_script = tmpl.format(
            final_model_pth,
            config_pth, 
            model, 
            resume_file, 
            save_dir, 
            task_name, 
            task_order, 
            replay_dir,
            str_mask_img,
            replay_prob
        )

        print("{}\n\n".format(run_script))
        # eval:
    exp_name = '{}_{}'.format(str_replay, replay_prob)
    export_f = None
    if kwargs.get("log", False):
        export_f = "> $ROOT/results/{}_run_{}_S{}.txt".format(cl_setting[0].upper(), exp_name, setting_idx)
    else:
        export_f = ""
    print("python -c \'from eval_os import *; stage_sweep(cl_setting=\"{}\", setting_idx={}, abbr_seq=\"{}\", device=\'${{DEVICE}}\', model_name=\"unicl\", save_dir=\"\'${{ROOT}}\'/exp/clvqa\", val_exp=\"{}\", test_stand_alone=False, test_reg=False)\' {}".format(cl_setting, setting_idx, arrv_seq, exp_name, export_f))


def gen_reg_seq(device, reg_type, cl_setting, abbrs, setting_idx=1, **kwargs):
    root_dir = kwargs.get("root_dir", "/Users/stan")
    print(f"ROOT={root_dir}")
    print(f"DEVICE={device}")
    prefix_tmpl = (
        "if [ ! -f \"{}\" ] ; then \n"
        " CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config={} \\\n"
        " model=unicl \\\n"
        " dataset=clvqa \\\n"
        " run_type=train_val \\\n"
        " env.save_dir={} \\\n"
        " training.trainer=CL \\\n"
        " training.checkpoint_interval=4000 \\\n"
        " training.CL.use_cl=True \\\n"
        " training.CL.use_icarl=False \\\n"
        " training.CL.use_callback=True \\\n"
        " training.CL.use_specific_optim=True \\\n"
        " training.CL.reg_type={} \\\n"
        " training.CL.reg_lambda=1.0 \\\n"
        " training.CL.cur_task={} \\\n"
        " training.CL.task_order={} \\\n"
        " training.CL.use_replay=False \\\n"
        " training.callbacks=[] \\\n"
        " optimizer.type=weight_reg_adamw \\\n"
        " training.batch_size=24 "
    )
    append_tmpl = (
        "\\\n"
        " checkpoint.resume_file={}/setting_{}_{}/{}/unicl_{}/unicl_final.pth \\\n"
        " training.CL.reg_params_pth={}/setting_{}_{}/{}/unicl_{}/models/reg_params.pkl "
    )
    task_order_arrv = abbrs
    stages = ABBR2TASKList(cl_setting=cl_setting, abbr_seq=abbrs)
    for idx, stage in enumerate(stages):
        final_model_pth = "$ROOT/exp/clvqa/save/{}/setting_{}_{}/{}/unicl_{}/unicl_final.pth".format(cl_setting, setting_idx, task_order_arrv, reg_type, stage)
        save_dir = "$ROOT/exp/clvqa/save/{}/setting_{}_{}/{}/unicl_{}".format(cl_setting, setting_idx, task_order_arrv, reg_type, stage) 
        pth_to_save = "$ROOT/exp/clvqa/save/{}".format(cl_setting)
        config_pth = "EXP_CONFIG/{}/cl_{}_unicl_standalone.yaml".format(cl_setting, stage)
        run_script = prefix_tmpl.format(
            final_model_pth,
             config_pth, 
             save_dir, 
             reg_type, 
             stage, 
             task_order_arrv
        )
        if idx>0:
            prev_stage = stages[idx-1]
            run_script += append_tmpl.format(
                pth_to_save, setting_idx, task_order_arrv, reg_type, prev_stage, 
                pth_to_save, setting_idx, task_order_arrv, reg_type, prev_stage
            )
        run_script += "\nfi"
        print("{}\n\n".format(run_script))
    
    export_f = None
    if kwargs.get("log", False):
        export_f = "> $ROOT/results/{}_run_{}_S{}.txt".format(cl_setting[0].upper(),reg_type,setting_idx)
    else:
        export_f = ""
    print("python -c \'from eval_os import *; stage_sweep(cl_setting=\"{}\", setting_idx={}, abbr_seq=\"{}\", device=\"\'${{DEVICE}}\'\", model_name=\"unicl\", save_dir=\"\'${{ROOT}}\'/exp/clvqa\", val_exp=\"{}\", test_stand_alone=False, test_reg=True)\' {}".format(cl_setting, setting_idx, task_order_arrv, reg_type, export_f))


def gen_restore_seq_with_ratio(device, cl_setting, abbrs, lm_model, ratio, use_gt=False, with_token=True, tr_perc=None, setting_idx=1, **kwargs):
    root_dir = kwargs.get("root_dir", "/Users/stan")
    print(f"ROOT={root_dir}")
    print(f"DEVICE={device}")
    tmpl = (
       "if [ ! -f \"{}\" ] ; then \n"
       " CUDA_VISIBLE_DEVICES=$DEVICE mmf_run config={} \\\n"
       " model=unicl \\\n"
       " dataset=clvqa \\\n"
       " training.CL.use_cl=True \\\n"
       " training.CL.use_callback=False \\\n"
       " training.CL.use_replay=True \\\n"
       " training.CL.replay_method=restore_with_prob \\\n"
       " training.CL.task_order={} \\\n"
       " training.CL.restore_rate={} \\\n"
       " training.CL.restore_dir=$ROOT/exp/clvqa/QAG_seq/{}/QAG_{}_{}/{}_replay/{}_{}_{} \\\n"
       " training.CL.restore_paths={} \\\n"
       " dataset_config.clvqa.use_mask_img=True \\\n"
       " dataset_config.clvqa.mask_img_prob=0.15 \\\n"
       " run_type=train_val \\\n"
       " checkpoint.resume_file={} \\\n"
       " env.save_dir={} \\\n"
       " training.checkpoint_interval=4000 \\\n"
       " training.callbacks=[] \n"
       "fi \n"
    )
    arrvs = abbrs
    stages = ABBR2TASKList(cl_setting=cl_setting, abbr_seq=abbrs)
    token_append = "task_token" if with_token else "wo_token"
    if tr_perc is not None:
        token_append += f"_tr{tr_perc}"
    use_gt_append = "use_gt" if use_gt else "not_use_gt"
    
    for idx,stage in enumerate(stages):
        if idx==0:
            continue
        stand_alone_dir = f"$ROOT/exp/clvqa/save/stand_alone/{cl_setting}"
        follow_dir = f"$ROOT/exp/clvqa/save/{cl_setting}"
        resume_file = "{}/unicl_{}/unicl_final.pth".format(stand_alone_dir,stages[idx-1]) if idx==1 else \
            "{}/setting_{}_{}/{}_replay_qag_seq_{}_{}_{}/unicl_{}/unicl_final.pth".format(
                follow_dir,
                setting_idx, arrvs, 
                lm_model, use_gt_append, token_append, ratio, 
                stages[idx-1]
            )
        restore_paths = ",".join(["{}_REPLAY[{}]_AT[{}].npy".format(arrvs, arrvs[i], arrvs[idx]) for i in range(idx)])
        final_model_pth = "$ROOT/exp/clvqa/save/{}/setting_{}_{}/{}_replay_qag_seq_{}_{}_{}/unicl_{}/unicl_final.pth".format(cl_setting, setting_idx, arrvs, lm_model, use_gt_append, token_append, ratio, stage)
        config_pth = "EXP_CONFIG/{}/cl_{}_unicl_standalone.yaml".format(cl_setting, stage)
        save_dir = "$ROOT/exp/clvqa/save/{}/setting_{}_{}/{}_replay_qag_seq_{}_{}_{}/unicl_{}".format(cl_setting,setting_idx, arrvs, lm_model, use_gt_append, token_append, ratio, stage)
        run_script = tmpl.format(
            final_model_pth,
            config_pth, 
            arrvs,
            ratio, 
            use_gt_append, cl_setting, token_append, lm_model, lm_model, cl_setting, arrvs, 
            restore_paths, 
            resume_file, 
            save_dir
        )
        print("{}\n\n".format(run_script))
    
    # eval:
    exp_name = '{}_replay_qag_seq_{}_{}_{}'.format(lm_model, use_gt_append, token_append, ratio)
    export_f = None
    if kwargs.get("log", False):
        export_f = "> $ROOT/results/{}_run_{}_S{}.txt".format(cl_setting[0].upper(),exp_name,setting_idx)
    else:
        export_f = ""
    print("python -c \'from eval_os import *; stage_sweep(cl_setting=\"{}\", setting_idx={}, abbr_seq=\"{}\", device=\"\'${{DEVICE}}\'\", model_name=\"unicl\", save_dir=\"\'${{ROOT}}\'/exp/clvqa\", val_exp=\"{}\", test_stand_alone=False, test_reg=False)\' {}".format(cl_setting, setting_idx, arrvs, exp_name, export_f))
    

def gen_task_seq(cl_setting, sample_n, load_pth=None):
    if load_pth is None:
        ori_task_seq = 'oarlks' if cl_setting == "functional" else "abcdef"

        all_permutations = np.array(list(itertools.permutations(ori_task_seq, len(ori_task_seq))))
        setting_indices = (np.arange(len(all_permutations)) + 1)


        sample_idx = np.random.choice(len(all_permutations)-1, sample_n, replace=False)
        sample_idx = np.sort(sample_idx)

        sampled_perm = all_permutations[1:][sample_idx].tolist()
        sampled_setting_idx = setting_indices[1:][sample_idx].tolist()

        sample_setting = "functional" if cl_setting == "functional" else "scene"
        with open(f"files/{cl_setting}_perm.pkl",'wb') as f:
            pkl.dump(
                dict(sampled_perm=sampled_perm, sampled_setting_idx=sampled_setting_idx),
                f,
                protocol=pkl.HIGHEST_PROTOCOL
            )
    else:
        assert os.path.isfile(load_pth)
        with open(load_pth,'rb') as f:
            perm = pkl.load(f)
        sampled_perm = perm['sampled_perm']
        sampled_setting_idx = perm['sampled_setting_idx']

    for idx, (perm, sid) in enumerate(zip(sampled_perm, sampled_setting_idx)):
        task_abbr = "".join(perm)
        
        # sequentially finetuning
        # os.system(f"python -c 'from generate_run_scripts import *; gen_ft_seq(4, \"{cl_setting}\", \"{task_abbr}\", {sid})' > mmf/run/m_scene/{cl_setting[0].upper()}_run_ft_seq_S{sid}.sh")
        
        # ewc
        # os.system(f"python -c 'from generate_run_scripts import *; gen_reg_seq(5, \"ewc\", \"{cl_setting}\", \"{task_abbr}\", {sid})' > mmf/run/m_scene/{cl_setting[0].upper()}_run_ewc_S{sid}.sh")
        
        # mas
        # os.system(f"python -c 'from generate_run_scripts import *; gen_reg_seq(6, \"mas\", \"{cl_setting}\", \"{task_abbr}\", {sid})' > mmf/run/m_scene/{cl_setting[0].upper()}_run_mas_S{sid}.sh")
        
        # rnd replay
        # for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
        #     os.system(f"python -c 'from generate_run_scripts import *; gen_random_replay_w_prob(0, cl_setting = \"{cl_setting}\", abbr_seq=\"{task_abbr}\", model=\"unicl\", replay_prob={r}, setting_idx={sid},replay_mask_img=True, use_gt_sg=False)' > run_scripts/{cl_setting[0].upper()}_run_rnd_wogt_mi_rp_{r}_S{sid}.sh")
        
        # kmeans rnd replay
        # os.system(f"python -c 'from generate_run_scripts import *; gen_kmeans_replay_w_prob(0, cl_setting = \"{cl_setting}\", abbr_seq=\"{task_abbr}\", model=\"unicl\", replay_prob=0.02, setting_idx={sid})' > mmf/run/m_scene/{cl_setting[0].upper()}_run_kmeans_rp_0.02_S{sid}.sh")
        
        # with SGP replayed samples
        for r in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5]:
            os.system(f"python -c 'from generate_run_scripts import *; gen_restore_seq_with_ratio(device=0, cl_setting=\"{cl_setting}\", abbrs=\"{task_abbr}\", lm_model=\"distilgpt2\", ratio={r}, use_gt=False, with_token=True, setting_idx={sid})' > run_scripts/mmclvqa/{cl_setting}/{cl_setting[0].upper()}_run_distilgpt2_wogt_wtt_{r}_S{sid}.sh")
        

def gen_task_s1(cl_setting, task_abbr):
    sid = 1
    for r in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5]:
        os.system(f"python -c 'from generate_run_scripts import *; gen_restore_seq_with_ratio(device=0, cl_setting=\"{cl_setting}\", abbrs=\"{task_abbr}\", lm_model=\"distilgpt2\", ratio={r}, use_gt=False, with_token=True, setting_idx={sid})' > run_scripts/mmclvqa/{cl_setting}/{cl_setting[0].upper()}_run_distilgpt2_wogt_wtt_{r}_S{sid}.sh")
        # os.system(f"python -c 'from generate_run_scripts import *; gen_random_replay_w_prob(0, cl_setting = \"{cl_setting}\", abbr_seq=\"{task_abbr}\", model=\"unicl\", replay_prob={r}, setting_idx={sid}, replay_mask_img=True, use_gt_sg=False)' > run_scripts/mmclvqa/{cl_setting[0].upper()}_run_rnd_mi_rp_{r}_S{sid}.sh")


if __name__=="__main__":
    gen_task_s1("functional", "oarlks")
    gen_task_seq("functional", 5, load_pth="files/functional_perm.pkl")
    gen_task_s1("scene", "abcdef")
    gen_task_seq("scene", 5, load_pth="files/scene_perm.pkl")

    # gen_random_replay_w_prob(6, cl_setting="scene", abbr_seq="abcdef", model="unicl", replay_prob = 0.015, replay_mask_img=False, setting_idx=1)
    # gen_kmeans_replay_w_prob(5, cl_setting="functional", abbr_seq="oarlks", model="unicl", replay_prob = 0.02, replay_mask_img=False, setting_idx=1)
    # gen_reg_seq(1, "ewc", cl_setting="scene", abbrs="abcdef")
    # gen_standalone(7, "scene", "unicl")
    # gen_ft_seq(3, "scene", "abcdef", setting_idx=1)

    # "python -c 'from generate_run_scripts import *; gen_random_replay_w_prob(0, cl_setting = \"{cl_setting}\", abbr_seq=\"{task_abbr}\", model=\"unicl\", replay_prob={r}, replay_mask_img=True, setting_idx={sid})' > mmf/run/mi/{cl_setting[0].upper()}_run_mi_rnd_rp_{r}_S{sid}.sh"
