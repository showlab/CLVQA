from ast import arg
import os
import json
from matplotlib import use
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
import re
import csv
import json
import uuid
import numpy as np

from mmf.common.CL_constant import GENERATED_SG_PTH, ABBR2TASK, TASK_DICT

def get_gen_token(task, specific=False):
    '''
    args: 
        task: task name of current stage
        specific: whether use task specific gen token 
    '''
    if specific:
        return "__" + task +  '__'
    else:
        return "__gen__"

def get_que_token(task, specific=False):
    '''
    args: 
        task: task name of current stage
        specific: whether use task specific que token 
    '''
    if specific:
        return f"[que_{task}]"
    else:
        return "[que]"


def get_cqa_context(model_arch, sg_prefix, gen_token, que_token, sep_token):
    if model_arch == "decoder-only":
        cqa_context = [gen_token + sep_token.join(sg_prefix) + que_token] # gpt2
    else:
        cqa_context = [que_token + sep_token.join(sg_prefix) + que_token] # t5-small
    
    return cqa_context



class GenUtilsDataset(Dataset):
    def __init__(self, cl_setting, task_abbv, gen_specific=False, que_specific=False, model_arch="decoder-only", use_gt=False, args=None):
        self.sep_token = "[SEP]"
        self.pad_token = "[pad]"
        self.eos_token = "<|endoftext|>"
        
        self.args = args
        self.cl_setting = cl_setting
        self.task = ABBR2TASK[cl_setting][task_abbv]
    
        assert model_arch in ['encoder-decoder','decoder-only']
        self.model_arch = model_arch
        
        self.use_gt_sg = use_gt
        self._init_SG_db(use_gt=use_gt)
        self._init_gen_token(gen_specific=gen_specific)
        self._init_que_token(que_specific=que_specific)
        
        # for not use gt
        self.n_sg_item = 1 if (self.task in ['object', 'attribute'] or self.cl_setting!="functional") else 2
    
    def _init_gen_token(self, gen_specific):
        self.gen_token = get_gen_token(task=self.task, specific=gen_specific)

    def _init_que_token(self, que_specific):
        self.que_token = get_que_token(task=self.task, specific=que_specific)

    
    def _init_SG_db(self, use_gt=False):
        if use_gt:
            anno_pth = TASK_DICT[self.cl_setting][self.task]['train']
            anno = np.load(anno_pth, allow_pickle=True)
            self.sg_db = []
            for datum in anno:
                sg_seq = datum['gt_scene_graph_seq']            # ['apple on table [SEP]', 'pencil in the box [SEP]', ...]
                sg_mask = datum['gt_scene_graph_mask']          # [0,1,...]
                sg_seq_rm_sep = [sg_item.split(self.sep_token)[0].strip() for sg_item in sg_seq]
                qa_related_indices = np.where(np.array(sg_mask)==1)[0]
                ctx_for_qa = np.array(sg_seq_rm_sep)[qa_related_indices].tolist()
                if len(ctx_for_qa) >= self.args.n_sg_seq:
                    ctx_for_qa = np.random.choice(ctx_for_qa, self.args.n_sg_seq, replace=False).tolist()
                self.sg_db.append(ctx_for_qa)
        else:
            path = GENERATED_SG_PTH[self.cl_setting]
            with open(path, "r") as f:
                sg_db = json.load(f)
            self.sg_db = sg_db[self.task]
        
        # re-init
        if self.args.change_train_size:
            pth = f"/Users/stan/code/functional_continual_learning_dev/SG_processing/generated_{self.cl_setting}_sg_all_stages_{self.args.train_perc}.json" 
            with open(pth, "r") as f:
                sg_db = json.load(f)
            self.sg_db = sg_db[self.task]


        if len(self.sg_db) > 15000:
            select_indices = np.random.choice(len(self.sg_db), 15000, replace=False)
            self.sg_db = np.array(self.sg_db)[select_indices].tolist()
            
        
        if self.args is not None:
            if self.args.gen_debug:
                self.sg_db = self.sg_db[:500]
    
    def __len__(self):
        return len(self.sg_db)
    
    def __getitem__(self, index):
        sg_prefix = self.sg_db[index]
        if not self.use_gt_sg:
            sg_prefix = sg_prefix[:self.n_sg_item]

        rtn_prefix = [self.sep_token.join(sg_prefix)]

        # sg genration prefix
        context = [self.gen_token + self.sep_token.join(sg_prefix) + self.sep_token]

        # qa generation prefix
        cqa_context = get_cqa_context(model_arch=self.model_arch, sg_prefix=sg_prefix, gen_token=self.gen_token, que_token=self.que_token, sep_token=self.sep_token)

        return dict(
            sg_prefix = rtn_prefix,
            context = context,
            cqa_context = cqa_context
        )


def text_collate_fn(batch):
    batch_keys = batch[0].keys()
    rtn_dict = {k:[] for k in batch_keys}
    for k in batch_keys:
        for item in batch:
            rtn_dict[k].append(item[k][0])
    
    return rtn_dict


def get_generation_dataloader(cl_setting, task_abbv, gen_specific=False, que_specific=False, model_arch="decoder-only", use_gt=False, args=None, batch_size=64, collate_fn=text_collate_fn) :
    dataset = GenUtilsDataset(cl_setting=cl_setting, task_abbv=task_abbv, gen_specific=gen_specific, que_specific=que_specific, model_arch=model_arch, use_gt=use_gt, args=args)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return dataloader


if __name__ == "__main__":

    from transformers import GPT2Tokenizer
    from tqdm import tqdm
    from easydict import EasyDict as edict
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_tokens(['[que]', '__gen__', '[SEP]'])
    tokenizer.add_special_tokens({"pad_token":"[pad]"})
    tokenizer.padding_side = 'left'
    args = edict(
        n_sg_seq=30,
        gen_debug=False,
    )

    for ta_ in 'oarlks':
        dl = get_generation_dataloader(
            cl_setting='functional', task_abbv=ta_, gen_specific=False, que_specific=False, model_arch='decoder-only', args=args, use_gt=False, batch_size=64
        )

        for batch in tqdm(dl,total=len(dl)):
            ctx = tokenizer.batch_encode_plus(batch['context'], padding=True, return_tensors='pt')
        


