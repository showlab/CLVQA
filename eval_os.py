import os
import numpy as np
import sys
import pprint
from copy import deepcopy
from mmf.common.CL_constant import TASK_DICT,  ABBR2TASK, N_TESTING_SAMPLES
from mmf.utils.m4c_evaluators import TextVQAAccuracyEvaluator

'''
Example for evaluation
CUDA_VISIBLE_DEVICES=0 mmf_run config=EXP_CONFIG/functional/cl_object_unicl_standalone.yaml \
    model=unicl \
    dataset=clvqa \
    run_type=val \
    env.save_dir=path_to_save_dir \
    checkpoint.resume_file=path_to/model_checkpoint.ckpt
'''

def isVanilla(cl_setting):
    if cl_setting in ["functional", "scene"]:
        return True
    return False

def ABBR2TASKList(cl_setting, abbr_seq):
    abbr_mapping = ABBR2TASK[cl_setting]
    taskList = [abbr_mapping[abbr] for abbr in abbr_seq]
    return taskList

def parse_acc(str_acc):
    acc = str_acc.split(" ")[-1]
    acc = float(acc)
    return acc

def measure_forgetting(acc_matrix, K):
    # measure forgetting after learning the K-th task:
    assert K <= len(acc_matrix)
    fetch_acc_matrix = acc_matrix[:K, :K-1]
    
    # f^k_j = max(l) a^l_j - a^k_j
    f_kj = np.max(fetch_acc_matrix[:K-1]) - fetch_acc_matrix[K-1]
    return f_kj
    
def check_log_file(path, value="val/clvqa/textvqa_accuracy"):
    if not os.path.isfile(path):
        return None
    
    with open(path, 'r') as f:
        for line in f.readlines():
            if value in line:
                split_record = line.split(',')
                rtn_dict = {}
                for s_ in split_record:
                    if "acc" in s_:
                        parse_acc = s_.split('/')[-1]
                        acc_key = parse_acc.split(":")[0].strip()
                        acc_value = parse_acc.split(":")[-1].strip()
                        acc_value = float(acc_value)
                        rtn_dict[acc_key] = acc_value
                return rtn_dict

    return None

def test_chance(cl_setting):
    evaluator = TextVQAAccuracyEvaluator()
    anno_dict = TASK_DICT[cl_setting]
    stage_2_MostFreqAns = dict()
    predlist = []
    for stage in anno_dict:
        stage_2_MostFreqAns[stage] = dict()
        # in training split
        train_anno_pth = anno_dict[stage]['train']
        train_anno = np.load(train_anno_pth, allow_pickle=True)
        for item in train_anno:
            for ans in item['answers']:
                stage_2_MostFreqAns[stage][ans] = stage_2_MostFreqAns[stage].get(ans, 0) + 1
        # rank each stage, find the most freq ans
        ans_arr = np.array([k for k,_ in stage_2_MostFreqAns[stage].items()])
        cnt_arr = np.array([v for _,v in stage_2_MostFreqAns[stage].items()])
        most_freq_ans = ans_arr[np.argmax(cnt_arr)]
        
        val_anno_pth = anno_dict[stage]['val']
        val_anno = np.load(val_anno_pth, allow_pickle=True)
        for item in val_anno:
            predlist.append({"pred_answer":most_freq_ans, "gt_answers":item["answers"]})
    final_acc = evaluator.eval_pred_list(predlist)
    print("Chance prediction, final acc is {}".format(final_acc))
    

def stage_sweep(cl_setting, setting_idx, abbr_seq, device, model_name, save_dir, val_exp, test_stand_alone=False, test_reg=False, report_metric=True, print_acc=False):
    '''
    model_name = "unicl"
    '''
    taskList = ABBR2TASKList(cl_setting=cl_setting, abbr_seq=abbr_seq)
    abbr_seq_simp = deepcopy(abbr_seq)
    if "scene" in cl_setting:
        abbr_seq = [task[:2] for task in taskList]
    else:
        abbr_seq = [task[0] for task in taskList]
    
    result = dict()
    resList = dict()
    setting_save_dir =  f"{save_dir}/save/{cl_setting}"
    stand_alone_save_dir = f"{save_dir}/save/stand_alone/{cl_setting}"
    stand_alone_val_dir = f"{save_dir}/save/stand_alone_val/{cl_setting}"
    
    for learning_idx in range(len(taskList)):
        learning_stage = taskList[learning_idx]
        learning_arrv = abbr_seq[learning_idx]

        for test_idx in range(len(taskList)):
            testing_stage = taskList[test_idx]
            testing_arrv = abbr_seq[test_idx]
            logfile = f"{setting_save_dir}/setting_{setting_idx}_{abbr_seq_simp}/val_{val_exp}/{learning_arrv}2{testing_arrv}/train.log" if (not test_stand_alone) or test_reg else \
                f"{stand_alone_val_dir}/{learning_arrv}2{testing_arrv}/train.log"

            rtn = check_log_file(logfile) if isVanilla(cl_setting) else check_log_file(logfile,"val/clvqa/textvqa_accuracy_cls/all_acc")
            if (rtn is not None):
                if (not isVanilla(cl_setting)) and len(rtn)==1:
                    pass
                else:
                    for k in rtn:
                        if result.get(k) is None: result[k] = dict()
                        if resList.get(k) is None: resList[k] = []
                        result[k][f"{learning_arrv}2{testing_arrv}"] = rtn[k]
                        resList[k].append((f"{learning_arrv}2{testing_arrv}", rtn[k]))
                    continue
            
            resume_path = f"{stand_alone_save_dir}/{model_name}_{learning_stage}/{model_name}_final.pth" if (learning_idx==0 or test_stand_alone) and (not test_reg) else \
                f"{setting_save_dir}/setting_{setting_idx}_{abbr_seq_simp}/{val_exp}/{model_name}_{learning_stage}/{model_name}_final.pth"
            
            logdir = f"{setting_save_dir}/setting_{setting_idx}_{abbr_seq_simp}/val_{val_exp}/{learning_arrv}2{testing_arrv}" if (not test_stand_alone) or test_reg else \
                f"{stand_alone_val_dir}/{learning_arrv}2{testing_arrv}"
            
            config_pth = f"EXP_CONFIG/{cl_setting}/cl_{testing_stage}_{model_name}_standalone.yaml"
            
            eval_cmd = (
                f"CUDA_VISIBLE_DEVICES={device} mmf_run config={config_pth} "
                f"model={model_name} "
                f"dataset=clvqa "
                f"run_type=val "
                f"env.save_dir={logdir} "
                f"checkpoint.resume_file={resume_path} "
                "training.callbacks=[]"
            )

            if not isVanilla(cl_setting):
                eval_cmd += " evaluation.metrics[0]=textvqa_accuracy_cls"
            
            print_cmd = f"Running command:\n {eval_cmd}"
            pprint.pprint(print_cmd)
            os.system(eval_cmd)
            rtn = check_log_file(logfile) if isVanilla(cl_setting) else check_log_file(logfile,"val/clvqa/textvqa_accuracy_cls/all_acc")
            assert rtn is not None
            
            for k in rtn:
                if result.get(k) is None: result[k] = dict()
                if resList.get(k) is None: resList[k] = []
                result[k][f"{learning_arrv}2{testing_arrv}"] = rtn[k]
                resList[k].append((f"{learning_arrv}2{testing_arrv}", rtn[k]))
    pprint.pprint(result)
    pprint.pprint(resList)
    
    result_a = None
    pra = None
    if report_metric:
        for k in resList:
            result_a = []
            for idx, task_ in enumerate(abbr_seq):
                L = resList[k][idx*len(abbr_seq): (idx+1) * len(abbr_seq)]
                task_acc_list = [item[1] for item in L]
                result_a.append(task_acc_list)
            result_a = np.array(result_a)
            n_task = len(result_a)
            
            acc = np.diagonal(result_a)   # diagonal of acc matrix
            fin = result_a[-1]            # the final step of the result
            
            weights = np.array([N_TESTING_SAMPLES[cl_setting][t[0]] for t in abbr_seq])
            # 1. avg acc
            fin_acc = fin
            # 2. backward transfer
            bwt = fin - acc
            # 3. forward transfer
            # TODO for fwt: need to calculate the baseline
            # 4. average forgetting
            forgetting = measure_forgetting(result_a, n_task)
            print(
                f"==> {k} | Final acc: {fin_acc.tolist()}, weight avg acc: {np.average(fin_acc, weights=weights)}. \n"
                f"==> {k} | Backward transfer: {bwt[:-1].tolist()}, weighted bwt: {np.average(bwt[:-1], weights=weights[:-1])} \n"
                f"==> {k} | Forgetting: {forgetting.tolist()}, weighted forgetting: {np.average(forgetting, weights=weights[:n_task-1])}."
            )
            if print_acc and ("textvqa" in k or "all" in k):
                pra = result_a
    
    if print_acc:
        pprint.pprint(np.transpose(pra))


def test_multi_task(device, cl_setting, model_name, save_dir):
    multi_task_model_path = f"{save_dir}/save/multitask/{model_name}_{cl_setting}_incremental/unicl_final.pth"
    assert os.path.isfile(multi_task_model_path)
    abbr =  list(ABBR2TASK[cl_setting].keys())
    stages = [ABBR2TASK[cl_setting][t] for t in abbr]
    resList = dict()
    for testing_idx in range(len(stages)):
        testing_stage = stages[testing_idx]
        testing_arrv = abbr[testing_idx]
        logfile = f"{save_dir}/save/multitask/val_{model_name}_{cl_setting}_incremental/multitask_2_{testing_arrv}/train.log"
        rtn = check_log_file(logfile)
        if rtn is not None:
            for k in rtn:
                if resList.get(k) is None: resList[k] = []
                resList[k].append((f"multitask_2_{testing_arrv}", rtn[k]))
            continue
        config_pth = f"EXP_CONFIG/{cl_setting}/cl_{testing_stage}_{model_name}_standalone.yaml"
        eval_cmd = (
            f"CUDA_VISIBLE_DEVICES={device} mmf_run config={config_pth} "
            f"model={model_name} "
            f"dataset=clvqa "
            f"run_type=val "
            f"env.save_dir={save_dir}/save/multitask/val_{model_name}_{cl_setting}_incremental/multitask_2_{testing_arrv} "
            f"checkpoint.resume_file={multi_task_model_path} "
            "training.callbacks=[] "
        )
        if not isVanilla(cl_setting):
            eval_cmd += " evaluation.metrics[0]=textvqa_accuracy_cls"       
        pprint.pprint(eval_cmd)
        os.system(eval_cmd)
        rtn = check_log_file(logfile)
        assert rtn is not None
        for k in rtn:
            if resList.get(k) is None: resList[k] = []
            resList[k].append((f"multitask_2_{testing_arrv}", rtn[k]))
    
    # for multitask_2_multitask
    logfile = f"{save_dir}/save/multitask/val_{model_name}_{cl_setting}_incremental/multitask_2_multitask/train.log"
    rtn = check_log_file(logfile)
    if rtn is not None:
        for k in rtn:
            if resList.get(k) is None: resList[k] = []
            resList[k].append(("multitask_2_multitask", rtn[k]))
    else:
        cfg_pth = f"EXP_CONFIG/{cl_setting}/cl_{cl_setting}_multitask_{model_name}.yaml"
        eval_cmd = (
            f"CUDA_VISIBLE_DEVICES={device} mmf_run config={cfg_pth} "
            f"model={model_name} " 
            f"dataset=clvqa "
            f"run_type=test "
            f"env.save_dir={save_dir}/save/multitask/val_{model_name}_{cl_setting}_incremental/multitask_2_multitask "
            f"checkpoint.resume_file={multi_task_model_path} "
            "training.callbacks=[] "
        )
        if not isVanilla(cl_setting):
            eval_cmd += " evaluation.metrics[0]=textvqa_accuracy_cls"   
        pprint.pprint(eval_cmd)
        os.system(eval_cmd)
        rtn = check_log_file(logfile)
        assert rtn is not None
        for k in rtn:
            if resList.get(k) is None: resList[k] = []
            resList[k].append(("multitask_2_multitask", rtn[k]))
    
    pprint.pprint(resList)
    

if __name__ =='__main__':
    # test_multi_task(device=2, cl_setting="scene", model_name='unicl', save_dir='/Users/stan/exp/clvqa')
    # stage_sweep(cl_setting='functional', setting_idx=1, abbr_seq='oarlks', device=0, model_name='unicl', save_dir='/Users/stan/exp/clvqa', val_exp='distilgpt2_replay_qag_seq_not_use_gt_task_token_1.5', test_stand_alone=False, test_reg=True, print_acc=False)
    # python -c 'from eval_os import *; stage_sweep(cl_setting="functional", setting_idx=505, abbr_seq="kaorls", device=0, model_name="unicl", save_dir="/Users/stan/exp/clvqa", val_exp="rnd_replay_0.02", test_stand_alone=False, test_reg=False)' > path_to_this_exp_result.txt
    # python -c 'from eval_os import *; stage_sweep(cl_setting="scene", setting_idx=194, abbr_seq="beacfd", device=1, model_name="unicl", save_dir="/Users/stan/exp/clvqa", val_exp="ft", test_stand_alone=False, test_reg=False)'
    test_chance("scene")