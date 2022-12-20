import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
import re
import csv
import json
import uuid
import numpy as np
from copy import deepcopy
import os
import logging
import pathlib
from collections import OrderedDict
from settings import args, TASK_DICT, SPECIAL_TOKENS, SPECIAL_TOKEN_IDS, FILL_VAL
from settings import TOKENIZER, LEN_FACTOR, DATA_ATTRS, MEMORY_FACTOR, MODEL_CONFIG, MODEL_CLASS
from mmf.datasets.gen_utils_dataset import get_gen_token, get_que_token
from multiprocessing import Pool
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
logger = logging.getLogger(__name__)

from functools import partial
tokenizer_encode = partial(TOKENIZER.encode, add_special_tokens=False)

class CLVQADataset(Dataset):
    def __init__(self, data_paths, data_type, extra_data=[]):
        self.data_type = data_type
        
        # directly use sep token here
        self.sep_token = SPECIAL_TOKEN_IDS["sep_token"]
        self.ans_token = SPECIAL_TOKEN_IDS["ans_token"]
        self.eos_token = SPECIAL_TOKEN_IDS["eos_token"]
        self.pad_token = SPECIAL_TOKEN_IDS["pad_token"]
        self.ocr_token = SPECIAL_TOKEN_IDS["ocr_token"]

        if not isinstance(data_paths, list):
            data_paths = [data_paths]
        annos = []
        for data_path in data_paths:
            if not data_path:
                continue
            anno = np.load(file=data_path, allow_pickle=True)
            annos.append(anno)  
        self.anno_db = np.concatenate(annos)
        
        if len(extra_data) > 0:
            extra_data = np.array(extra_data)
            self.anno_db = np.concatenate([self.anno_db, extra_data])
        
        if args.debug:
            self.anno_db = self.anno_db[:100]
        
        if args.change_train_size:
            annos = []
            for data_path in data_paths:
                basename = os.path.basename(data_path)
                f = f"/Users/stan/code/functional_continual_learning_dev/Gen_data/ds_function/{args.train_perc}/{basename}"
                annos.append(np.load(f, allow_pickle=True))
            self.anno_db = np.concatenate(annos)

            if len(extra_data) > 0:
                extra_data = np.random.choice(extra_data, int(args.train_perc*len(extra_data)), replace=False)
                self.anno_db = np.concatenate([self.anno_db, extra_data])


    def concat_example(self, gen_token, c, question_token, q, ans_token, a, eos_token):
        example = question_token + q + ans_token + a
        if len(example) + 1 > args.max_len:
            logger.warning('an example with len {} is too long!'.format(len(example) + 1))
            return
        example = gen_token + c[:args.max_len-len(example)-1] + example + eos_token
        return example

    def parse_example(
        self, 
        gen_token,
        que_token,
        ctx_for_qa,
        context, 
        question, 
        answer, 
        idx, 
    ):
        
        if args.model_arch == "decoder-only":
            # args for concat_example: gen_token, context, question_token, question, answer_token, answer, eos_token
            # c_example for testing: given gen_token and context, generate question and answer
            c_example = self.concat_example([gen_token], ctx_for_qa, [que_token], [], [], [], [])
            
            cqa_example = self.concat_example([gen_token], ctx_for_qa, [que_token], question, [self.ans_token], answer, [])
            Y_example = self.concat_example([], [], [], question, [self.ans_token], answer, [self.eos_token])
            Y_example = [FILL_VAL] * (len(cqa_example) - len(Y_example)) + Y_example
            
            gen_X_example = self.concat_example([gen_token], context, [], [], [], [], [])
            gen_Y_example = self.concat_example([], context, [], [], [], [], [self.eos_token])
        
        else: # model_arch is "encoder-decoder" 
            c_example = self.concat_example([que_token], ctx_for_qa, [que_token], [], [], [], [])
            cqa_example = self.concat_example([que_token], ctx_for_qa, [que_token], [], [], [], [])
            Y_example = self.concat_example([], [], [], question, [self.ans_token], answer, [self.eos_token])
            
            # devide context into input and targets
            np_context = np.array(context)
            sep_indices = np.where(np_context==self.sep_token)[0]
            if len(sep_indices)>0:
                split_point = np.random.choice(sep_indices)
                context_inp = np_context[:split_point+1].tolist()
                context_outp = np_context[split_point+1:].tolist()
            else:
                context_inp = context + [self.sep_token]
                context_outp = []
            gen_X_example = self.concat_example([gen_token], context_inp, [], [], [], [], [])
            gen_Y_example = self.concat_example([], context_outp, [], [], [], [], [self.eos_token])

        return {
            "context": c_example,
            "len_context": len(c_example),
            "cqa": cqa_example,
            "len_cqa": len(cqa_example),
            "Y": Y_example,
            "gen_X": gen_X_example,
            "gen_Y": gen_Y_example,
            "qid": idx,         
        }

    def tokenization(self, datum):
        sg_seq = datum['gt_scene_graph_seq']            # ['apple on table [SEP]', 'pencil in the box [SEP]', ...]
        sg_mask = datum['gt_scene_graph_mask']          # [0,1,...]
        sg_seq_rm_sep = [sg_item.split(SPECIAL_TOKENS['sep_token'])[0].strip() for sg_item in sg_seq]
        
        qa_related_indices = np.where(np.array(sg_mask)==1)[0]
        qa_nonrelated_indices = np.where(np.array(sg_mask)==0)[0]

        ctx_for_qa = np.array(sg_seq_rm_sep)[qa_related_indices].tolist()
        ctx_for_nonqa = np.array(sg_seq_rm_sep)[qa_nonrelated_indices].tolist()
        if len(ctx_for_qa) >= args.n_sg_seq:
            ctx_for_qa = np.random.choice(ctx_for_qa, args.n_sg_seq, replace=False).tolist()
        elif len(ctx_for_nonqa) > (args.n_sg_seq - len(ctx_for_qa)):
            ctx_for_nonqa = np.random.choice(ctx_for_nonqa, args.n_sg_seq - len(ctx_for_qa), replace=False).tolist()
        
        context = ctx_for_qa + ctx_for_nonqa

        ctx_for_qa = SPECIAL_TOKENS['sep_token'].join(ctx_for_qa)
        ctx_for_qa = tokenizer_encode(ctx_for_qa)
        
        context = SPECIAL_TOKENS['sep_token'].join(context)
        context = tokenizer_encode(context)
        
        question = datum['question']
        question = tokenizer_encode(question)
  
        # choose one of the answer for generation
        answer = np.random.choice(datum['answers'])
        answer = tokenizer_encode(answer)
        
        qid = datum['question_id']
        stage_in_anno = datum['stage']
        stage_name = stage_in_anno
        if "gen" in stage_in_anno:
            stage_name = stage_in_anno.split("_")[0]
        

        arg_gen_token = SPECIAL_TOKEN_IDS[stage_name]
        arg_que_token = SPECIAL_TOKEN_IDS[f"que_{stage_name}"]

        return self.parse_example(
                gen_token=arg_gen_token,
                que_token=arg_que_token,
                ctx_for_qa=ctx_for_qa,
                context=context, 
                question=question, 
                answer=answer, 
                idx=qid, 
            )


    def sort(self):
        self.data.sort(key=lambda x: len(x["cqa"]))
        return self

    def sort_by_index(self):
        self.data.sort(key=lambda x: x["qid"])

    def get_indices(self):
        return [d["qid"] for d in self.data]

    def __len__(self):
        return len(self.anno_db)

    def __getitem__(self, index):
        datum = self.anno_db[index]
        rtn_parse = self.tokenization(datum=datum)
        return rtn_parse
        