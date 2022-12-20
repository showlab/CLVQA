import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
import re
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
from mmf.datasets.gen_utils_dataset import get_generation_dataloader
from multiprocessing import Pool
from tqdm import tqdm
import sys
import time
import quadprog
import io

from functools import partial
tokenizer_encode = partial(TOKENIZER.encode, add_special_tokens=False)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
logger = logging.getLogger(__name__)


def make_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


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


def get_model_dir(tasks):
    return os.path.join(args.model_dir_root, tasks[0]) if args.seq_train_type != "multitask" else args.model_dir_root


def get_losses(parallel_model, cqa, cqa_token_type_id, Y, gen_X, gen_X_token_type_id, gen_Y, loss_fct):
    if "lll" in args.seq_train_type:
        qa_logits = parallel_model(
            [
                (dict(input_ids=cqa[i], token_type_ids=cqa_token_type_id[i]),) for i in range(len(cqa))
            ]
        )
        lm_logits = parallel_model(
            [
                (dict(input_ids=gen_X[i], token_type_ids=gen_X_token_type_id[i]),) for i in range(len(gen_X))
            ]
        )
        qa_loss = loss_fct([torch.transpose(l, 1, 2) for l in qa_logits], Y)
        lm_loss = loss_fct([torch.transpose(l, 1, 2) for l in lm_logits], gen_Y)
        return torch.mean(qa_loss), args.lm_lambda * torch.mean(lm_loss)
    else:
        qa_logits = parallel_model(cqa)
        qa_loss = loss_fct([torch.transpose(l, 1, 2) for l in qa_logits], Y)
        return torch.mean(qa_loss), torch.tensor(0.)

def get_losses_wo_tkt(parallel_model, cqa,  Y, gen_X, gen_Y, loss_fct):
    '''
    get lossed without token type ids
    '''
    if "lll" in args.seq_train_type:
        if args.model_arch == "decoder-only":
            qa_logits = parallel_model(
                [
                    (dict(input_ids=cqa[i],),) for i in range(len(cqa))
                ]
            )
            lm_logits = parallel_model(
                [
                    (dict(input_ids=gen_X[i],),) for i in range(len(gen_X))
                ]
            )
            qa_loss = loss_fct([torch.transpose(l, 1, 2) for l in qa_logits], Y)
            lm_loss = loss_fct([torch.transpose(l, 1, 2) for l in lm_logits], gen_Y)
        elif args.model_arch == "encoder-decoder":
            qa_loss = parallel_model(
                [
                    (dict(input_ids=cqa[i], labels=Y[i]),) for i in range(len(gen_X))
                ]
            )[0]
            lm_loss = parallel_model(
                [
                    (dict(input_ids=gen_X[i], labels=gen_Y[i]),) for i in range(len(gen_X))
                ]
            )[0]
        else:
            raise NotImplementedError
        return torch.mean(qa_loss), args.lm_lambda * torch.mean(lm_loss)

    else:
        qa_logits = parallel_model(cqa)
        qa_loss = loss_fct([torch.transpose(l, 1, 2) for l in qa_logits], Y)
        return torch.mean(qa_loss), torch.tensor(0.)

def pad_to_max_len(l, pad_len, val):
    return l + [val] * pad_len


def pad_all_to_max_len(ls, val):
    max_len = max(len(l) for l in ls)
    return [pad_to_max_len(l, max_len-len(l), val) for l in ls]


def varlen_collate_fn(data):
    batch_size = (len(data) + args.n_gpus - 1) // args.n_gpus
    cqs = torch.tensor(pad_all_to_max_len([datum[0] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    len_cqs = torch.tensor([datum[1] for datum in data]).split(batch_size)
    cqas = torch.tensor(pad_all_to_max_len([datum[2] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    len_cqas = torch.tensor([datum[3] for datum in data]).split(batch_size)
    Ys = torch.tensor(pad_all_to_max_len([datum[4] for datum in data], FILL_VAL)).split(batch_size)
    gen_Xs = torch.tensor(pad_all_to_max_len([datum[5] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    gen_Ys = torch.tensor(pad_all_to_max_len([datum[6] for datum in data], FILL_VAL)).split(batch_size)
    return list(cqs), list(len_cqs), list(cqas), list(len_cqas), list(Ys), list(gen_Xs), list(gen_Ys)


def dynamic_collate_fn(data, batch_size):
    def local_collate():
        null_counter = 0
        _cs, _len_cs, _cqas, _len_cqas, _Ys, _gen_Xs, _gen_Ys = [], [], [], [], [], [], []
        _c_tks, _cqa_tks, _gen_X_tks = [], [], []
        
        Y_max_len = max(len(data[j]['Y']) for j in range(st, ed))
        c_max_len = max(len(data[j]['context']) for j in range(st, ed))
        
        for j in range(st, ed):
            if None in data[j] or len(data[j])==0:
                null_counter+=1
                logger.warning('null example in collate_fn, count: {}'.format(null_counter))
                continue

            pad_len = cqa_max_len - len(data[j]['cqa'])

            _cs.append(pad_to_max_len(data[j]['context'], c_max_len-len(data[j]['context']), SPECIAL_TOKEN_IDS["pad_token"]))
            _c_tks.append(pad_to_max_len(data[j]['context_token_type'], c_max_len-len(data[j]['context_token_type']), 0))
            _len_cs.append(data[j]['len_context'])
            
            _cqas.append(pad_to_max_len(data[j]['cqa'], pad_len, SPECIAL_TOKEN_IDS["pad_token"]))
            _cqa_tks.append(pad_to_max_len(data[j]['cqa_token_type'], pad_len, 0))
            _len_cqas.append(data[j]['len_cqa'])
            _Ys.append(pad_to_max_len(data[j]['Y'], Y_max_len - len(data[j]['Y']), FILL_VAL))
            
            _gen_Xs.append(pad_to_max_len(data[j]['gen_X'], pad_len, SPECIAL_TOKEN_IDS["pad_token"]))
            _gen_X_tks.append(pad_to_max_len(data[j]['gen_X_token_type'], pad_len, 0))
            _gen_Ys.append(pad_to_max_len(data[j]['gen_Y'], pad_len, FILL_VAL))

        cqs.append(torch.tensor(_cs))
        c_tks.append(torch.tensor(_c_tks))
        len_cqs.append(torch.tensor(_len_cs))

        cqas.append(torch.tensor(_cqas))
        cqa_tks.append(torch.tensor(_cqa_tks))
        len_cqas.append(torch.tensor(_len_cqas))
        Ys.append(torch.tensor(_Ys))

        gen_Xs.append(torch.tensor(_gen_Xs))
        gen_X_tks.append(torch.tensor(_gen_X_tks))
        gen_Ys.append(torch.tensor(_gen_Ys))

    cqs, len_cqs, cqas, len_cqas, Ys, gen_Xs, gen_Ys = [], [], [], [], [], [], []
    c_tks, cqa_tks, gen_X_tks = [], [], []
    cqa_max_len, cnt, st = 0, 0, 0
    for ed, datum in enumerate(data):
        ln = len(datum['cqa']) # use cqas to calibrate
        if max(cqa_max_len, ln)**LEN_FACTOR * (ed - st + 1) > batch_size[cnt]:
            local_collate()
            cnt += 1
            if cnt == args.n_gpus:
                return {
                    "contexts": cqs,
                    "contexts_token_type_ids": c_tks,
                    "len_contexts": len_cqs,
                    "cqas": cqas,
                    "cqa_token_type_ids": cqa_tks,
                    "len_cqas": len_cqas,
                    "Ys": Ys,
                    "gen_Xs": gen_Xs,
                    "gen_X_token_type_ids": gen_X_tks,
                    "gen_Ys": gen_Ys
                }
            cqa_max_len = 0
            st = ed
        cqa_max_len = max(cqa_max_len, ln)
    ed += 1  # otherwise ed will be len(data)-1
    local_collate()

    return {
        "contexts": cqs,
        "contexts_token_type_ids": c_tks,
        "len_contexts": len_cqs,
        "cqas": cqas,
        "cqa_token_type_ids": cqa_tks,
        "len_cqas": len_cqas,
        "Ys": Ys,
        "gen_Xs": gen_Xs,
        "gen_X_token_type_ids": gen_X_tks,
        "gen_Ys": gen_Ys
    }


def dynamic_collate_wo_tkt(data, batch_size):
    def local_collate():
        null_counter = 0
        _cs, _len_cs, _cqas, _len_cqas, _Ys, _gen_Xs, _gen_Ys = [], [], [], [], [], [], []
        
        gen_X_max_len = max(len(data[j]['gen_X']) for j in range(st, ed))
        gen_Y_max_len = max(len(data[j]['gen_Y']) for j in range(st, ed))
        Y_max_len = max(len(data[j]['Y']) for j in range(st, ed))
        c_max_len = max(len(data[j]['context']) for j in range(st, ed))

        
        for j in range(st, ed):
            if None in data[j] or len(data[j])==0:
                null_counter+=1
                logger.warning('null example in collate_fn, count: {}'.format(null_counter))
                continue

            pad_len = cqa_max_len - len(data[j]['cqa'])

            _cs.append(pad_to_max_len(data[j]['context'], c_max_len-len(data[j]['context']), SPECIAL_TOKEN_IDS["pad_token"]))
            _len_cs.append(data[j]['len_context'])
            
            _cqas.append(pad_to_max_len(data[j]['cqa'], pad_len, SPECIAL_TOKEN_IDS["pad_token"]))
            _len_cqas.append(data[j]['len_cqa'])
            _Ys.append(pad_to_max_len(data[j]['Y'], Y_max_len - len(data[j]['Y']), FILL_VAL))
            
            _gen_Xs.append(pad_to_max_len(data[j]['gen_X'], gen_X_max_len - len(data[j]['gen_X']), SPECIAL_TOKEN_IDS["pad_token"]))
            _gen_Ys.append(pad_to_max_len(data[j]['gen_Y'], gen_Y_max_len - len(data[j]['gen_Y']), FILL_VAL))

        cqs.append(torch.tensor(_cs))
        len_cqs.append(torch.tensor(_len_cs))

        cqas.append(torch.tensor(_cqas))
        len_cqas.append(torch.tensor(_len_cqas))
        Ys.append(torch.tensor(_Ys))

        gen_Xs.append(torch.tensor(_gen_Xs))
        gen_Ys.append(torch.tensor(_gen_Ys))

    cqs, len_cqs, cqas, len_cqas, Ys, gen_Xs, gen_Ys = [], [], [], [], [], [], []
    cqa_max_len, cnt, st = 0, 0, 0
    for ed, datum in enumerate(data):
        ln = len(datum['cqa']) # use cqas to calibrate
        if max(cqa_max_len, ln)**LEN_FACTOR * (ed - st + 1) > batch_size[cnt]:
            local_collate()
            cnt += 1
            if cnt == args.n_gpus:
                return {
                    "contexts": cqs,
                    "len_contexts": len_cqs,
                    "cqas": cqas,
                    "len_cqas": len_cqas,
                    "Ys": Ys,
                    "gen_Xs": gen_Xs,
                    "gen_Ys": gen_Ys
                }
            cqa_max_len = 0
            st = ed
        cqa_max_len = max(cqa_max_len, ln)
    ed += 1  # otherwise ed will be len(data)-1
    local_collate()

    return {
        "contexts": cqs,
        "len_contexts": len_cqs,
        "cqas": cqas,
        "len_cqas": len_cqas,
        "Ys": Ys,
        "gen_Xs": gen_Xs,
        "gen_Ys": gen_Ys
    }


class QADataset(Dataset):
    def __init__(self, data_paths, data_type, gen_token, extra_data=[]):
        self.data_type = data_type
        self.gen_token = gen_token
        
        # directly use sep token here
        self.sep_token = SPECIAL_TOKEN_IDS["sep_token"]
        self.question_token = SPECIAL_TOKEN_IDS['question_token']
        self.ans_token = SPECIAL_TOKEN_IDS["ans_token"]
        self.eos_token = SPECIAL_TOKEN_IDS["eos_token"]
        self.pad_token = SPECIAL_TOKEN_IDS["pad_token"]
        self.ocr_token = SPECIAL_TOKEN_IDS["ocr_token"]

        if not isinstance(data_paths, list):
            data_paths = [data_paths]

        data = []
        self.raw_anno = []
        for data_path in data_paths:
            if not data_path:
                continue
            raw_anno = np.load(file=data_path, allow_pickle=True)
            d = []
            for raw_d in raw_anno:
                scene_graphs = raw_d['gt_scene_graph_seq']       # is a list
                scene_graphs_mask = raw_d['gt_scene_graph_mask'] # is a list
                question = raw_d['question']                     # str
                answers = raw_d['answers']
                question_id = raw_d['question_id']
                d.append(
                    dict(
                        scene_graphs_list = scene_graphs, 
                        scene_graphs_mask = scene_graphs_mask, 
                        question = question, 
                        answers = answers, 
                        question_id = question_id)
                )
            data += d
            self.raw_anno.extend(raw_anno)
        
        self.data = []
        self.max_a_len = 0

        if len(data) > 0:
            self.data_tokenization(data) # list[ dict ]

        # TODO: fix here for extra data
        if len(extra_data) > 0:
            extra_data = map(lambda x: self.etl_single_extra_data(x), extra_data)
            extra_data = list(filter(lambda x:x, extra_data))
            if args.gen_lm_sample_percentage > 0. and len(extra_data) == 0:
                logger.warning("No good extra data but sample percentage > 0!")
            self.data += extra_data

    def etl_single_extra_data(self, data):
        gen_token = data[0]
        data = ' '.join([str(datum) for datum in data[1:]])
        try:
            if args.use_sep:
                context, qa = re.split(str(SPECIAL_TOKEN_IDS["sep_token"]), data)
            else:
                context = ""
                qa = data
            question, answer = re.split(str(SPECIAL_TOKEN_IDS["ans_token"]), qa)
            context = [int(c) for c in context.strip().split()]
            question = [int(q) for q in question.strip().split()]
            answer = [int(a) for a in re.sub(str(SPECIAL_TOKEN_IDS["eos_token"]), "", answer).strip().split()]
            uid = uuid.uuid1().hex
            data = self.parse_example(gen_token, context, question, answer, uid)
        except ValueError:
            return
        return data

    def concat_example(self, gen_token, c, question_token, q, ans_token, a, eos_token):
        example = question_token + q + ans_token + a
        if len(example) + 1 > args.max_len:
            logger.warning('an example with len {} is too long!'.format(len(example) + 1))
            return
        example = gen_token + c[:args.max_len-len(example)-1] + example + eos_token
        return example

    def parse_example(self, gen_token, context, question, answer, idx, context_token_type=None, question_token_type=None, answer_token_type=None):
        # args for concat_example: gen_token, context, question_token, question, answer_token, answer, eos_token
        
        # c_example for testing: given gen_token and context, generate question and answer
        c_example = self.concat_example([gen_token], context, [self.question_token], [], [], [], [])
        c_token_type = self.concat_example([1,], context_token_type, [1,], [], [], [], [])
        
        cqa_example = self.concat_example([gen_token], context, [self.question_token], question, [self.ans_token], answer, [])
        cqa_token_type = self.concat_example([1,], context_token_type, [1,], question_token_type, [1,], answer_token_type, [])

        Y_example = self.concat_example([], [], [self.question_token], question, [self.ans_token], answer, [self.eos_token])
        Y_example = [FILL_VAL] * (len(cqa_example) - len(Y_example)) + Y_example
        
        gen_X_example = self.concat_example([gen_token], context, [self.question_token], question, [self.ans_token], answer, [])
        gen_X_token_type = self.concat_example([1,], context_token_type, [1,], question_token_type, [1,], answer_token_type, [])
        gen_Y_example = self.concat_example([], context, [self.question_token], question, [self.ans_token], answer, [self.eos_token])
        
        return {
            "context": c_example,
            "context_token_type": c_token_type,
            "len_context": len(c_example),
            "cqa": cqa_example,
            "cqa_token_type": cqa_token_type,
            "len_cqa": len(cqa_example),
            "Y": Y_example,
            "gen_X": gen_X_example,
            "gen_X_token_type": gen_X_token_type,
            "gen_Y": gen_Y_example,
            "qid": idx,            
        }

    def parallel_tokenization(self, d):
        examples  = []
        max_a_len = 0
        context_mask = []                           # for elements used for generating questions
        sg_seq = d['scene_graphs_list']            # ['apple on table [SEP]', 'pencil in the box [SEP]', ...]
        sg_mask = d['scene_graphs_mask']          # [0,1,...]
        sg_seq_rm_sep = [sg_item.split(SPECIAL_TOKENS['sep_token'])[0].strip() for sg_item in sg_seq]
        context = SPECIAL_TOKENS['sep_token'].join(sg_seq_rm_sep)
        context = tokenizer_encode(context)
        # mask for after tokenization
        cursor = 0
        for cid in context:
            context_mask.append(sg_mask[cursor])
            if cid == self.sep_token:
                cursor+=1
        assert len(context)==len(context_mask)
        
        question = d['question']
        question = tokenizer_encode(question)
        question_mask = [1,] * len(question)
        
        # choose one of the answer for generation
        answer = np.random.choice(d['answers'])
        answer = tokenizer_encode(answer)
        answer_mask = [1,] * len(answer)
        max_a_len = max(max_a_len, len(answer))
        
        qid = d['question_id']
        
        # TODO fix parse_example
        # should handle mask for the whold sequence, which is not implemented by original LAMOL
        examples.append(
            self.parse_example(
                gen_token=self.gen_token, 
                context=context, 
                question=question, 
                answer=answer, 
                idx=qid, 
                context_token_type=context_mask, 
                question_token_type=question_mask, 
                answer_token_type=answer_mask
            )
        )
        return examples, max_a_len

    def data_tokenization(self, data):
        if args.debug:
            data = data[:10]
            new_data = []
            for datum in data:
                new_data.append(self.parallel_tokenization(datum))
            data = new_data
        else:
            with Pool(args.n_workers) as pool:
                data = pool.map(self.parallel_tokenization, data)
        
        # After processing, init self.data
        for datum, max_a_len in data:
            self.data.extend(datum)
            self.max_a_len = max(self.max_a_len, max_a_len)

    def sort(self):
        self.data.sort(key=lambda x: len(x["cqa"]))
        return self

    def sort_by_index(self):
        self.data.sort(key=lambda x: x["qid"])

    def get_indices(self):
        return [d["qid"] for d in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class EarlyStopping:
    def __init__(self, logger,  patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.logger = logger

    def __call__(self, val_loss, model, model_dir):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_dir)
        elif score < self.best_score:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_dir)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_dir):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save_pretrained(model_dir)
        TOKENIZER.save_pretrained(model_dir)
        self.val_loss_min = val_loss

class TrainStep:
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __call__(self, loss, scheduler_steps):
        if not args.fp32:
            self.optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

        if not args.fp32:
            self.optimizer.update_master_grads()
            self.optimizer.clip_master_grads(args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

        if "gem" in args.seq_train_type and self.model.task_id >0: 
            store_grad(self.model.parameters, self.model.grads, self.model.grad_dims,self.model.task_id)
            indx = torch.cuda.LongTensor([i for i in range(self.model.task_id)])
            dotp = torch.mm(self.model.grads[:, self.model.task_id].unsqueeze(0),
                            self.model.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.model.grads[:, self.model.task_id].unsqueeze(1),
                              self.model.grads.index_select(1, indx), args.qp_margin)
                # copy gradients back
                overwrite_grad(self.model.parameters,
                               self.model.grads[:, self.model.task_id],
                               self.model.grad_dims)
            
        if args.seq_train_type in args.REG_TYPE_KEYS: # ewc / mas
            self.optimizer.step(self.model.reg_params)
        else:
            self.optimizer.step()
        if args.fp32 or (not self.optimizer.overflow):
            for i in range(scheduler_steps):
                self.scheduler.step()
        self.optimizer.zero_grad()



class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, data_type, max_batch_size):
        self.dataset = dataset
        self.data_type = data_type
        if data_type == "train":
            self.batch_size = args.train_batch_size
        else:
            self.batch_size = args.test_batch_size
        self.n_samples = len(dataset)
        self.max_batch_size = max_batch_size

    def __iter__(self):
        if args.debug or self.data_type == "test":
            indices = range(self.n_samples)
        else:
            indices = np.random.permutation(self.n_samples)
        max_len, cnt, st = 0, 0, 0
        batch = []
        for ed, idx in enumerate(indices):
            ln = max(len(self.dataset[idx]["cqa"]), len(self.dataset[idx]['gen_X']))
            if max(max_len, ln)**LEN_FACTOR * (ed - st + 1) > self.batch_size[cnt]:
                st = ed
                cnt += 1
                max_len = 0
                if cnt == args.n_gpus:
                    yield batch
                    cnt = 0
                    batch = []
            max_len = max(max_len, ln)
            batch.append(idx)
            if len(batch) == self.max_batch_size and self.data_type == "train":
                yield batch
                cnt, max_len, st = 0, 0, ed
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        raise NotImplementedError

def create_dataloader(dataset, data_type, max_batch_size=1000000000):
    if data_type == "train":
        batch_size = args.train_batch_size
    else:
        batch_size = args.test_batch_size

    if isinstance(batch_size, list):
        collate_fn=lambda x,bs=batch_size: dynamic_collate_wo_tkt(x, bs)
        shuffle = False
        batch_size = 1
        batch_sampler = DynamicBatchSampler(dataset, data_type, max_batch_size)
    else:
        collate_fn=lambda x: varlen_collate_fn(x)
        shuffle = not (data_type != "train" or args.debug)
        batch_sampler = None

    dataloader =  DataLoader(dataset, num_workers=args.n_workers,
                             collate_fn=collate_fn,
                             shuffle=shuffle,
                             batch_size=batch_size,
                             batch_sampler=batch_sampler)
    return dataloader

class WrapModel(torch.nn.Module):
    def __init__(self, model):
        super(WrapModel, self).__init__()
        self.model = model

    def forward(self, input):
        outputs = self.model(**input)
        return outputs[0]
 

def store_grad(get_ps, grads, grad_dims, task_id): 
    i = 0
    for param in get_ps():
        if param.grad is not None:
            beg = 0 if i == 0 else sum(grad_dims[:i])
            end = sum(grad_dims[:i+1])
            grads[beg: end, task_id].copy_(param.grad.data.view(-1))
        i += 1

def overwrite_grad(pp, newgrad, grad_dims):
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1

def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))

# CLVQA specific functions
def fcl_create_data(taskList, cur_task, model, sg_dta_base_path, special_tokens, special_token_ids, batch_size=32, use_save=True):
    sg_database = json.load(open(sg_dta_base_path,'r'))
    order = "".join([t[0] for t in taskList])
    cur_task_idx = taskList.index(cur_task)
    _all_gen_pseudo_training_samples = []
    rtn_for_train = []
    model.eval()
    for i in range(cur_task_idx):
        prev_task = taskList[i]
        stage_sg = sg_database[prev_task]
        gen_token = get_gen_token(task=prev_task, specific=args.add_task_tokens) # or special_tokens[prev_task]
        # n_sg_item = 1 if (prev_task in ['object','attribute']) or (args.cl_setting == "scene") else 2
        tot = len(stage_sg)
        
        task_dataloader = get_generation_dataloader(
            cl_setting=args.cl_setting, 
            task_abbv=prev_task[0], 
            gen_specific=args.add_task_tokens,
            que_specific=args.add_task_tokens and args.model_arch == "encoder-decoder",
            model_arch=args.model_arch,
            use_gt=args.use_gt,
            args=args,
            batch_size=batch_size
        )
        SGList = []
        QueList = []
        AnsList = []
        N_GT_ST_List = []
        for batch_idx, batch in tqdm(enumerate(task_dataloader), total=len(task_dataloader)):
            # (1) first generate scene graph, (2) based on the generated scene graph, generate QA 
            # gen_sg_ids=generate_text_ids, cqa_ids=generate_qa_ids
            # ed = st + batch_size if (st+batch_size)<tot else tot
            # batch_sg = stage_sg[st:ed]
            # batch_prefix = [sgs[:n_sg_item] for sgs in batch_sg]
            sg_prefix, context, cqa_context = batch['sg_prefix'], batch['context'], batch['cqa_context']
            rtn = model_generate_sg_batch(
                model = model, 
                tokenizer=TOKENIZER, 
                context=context,
                cqa_context=cqa_context, 
                sep_token=special_tokens['sep_token'], 
                gen_token=gen_token,
                que_token=special_tokens[f'que_{prev_task}'],
                args=args
            )
            for idx, (rtn_sg_ids, rtn_cqa_ids) in enumerate(zip(rtn['gen_sg_ids'].tolist(), rtn['cqa_ids'].tolist())):
                if not Check_Sanity(                            # if not legal, skip
                    sg_ids=rtn_sg_ids,
                    cqa_ids=rtn_cqa_ids, 
                    sep_token_id=special_token_ids['sep_token'],
                    que_token_id=special_token_ids[f'que_{prev_task}'],
                    ans_token_id=special_token_ids['ans_token'],
                    eos_token_id=special_token_ids['eos_token'],
                    model_arch=args.model_arch
                ):
                    continue

                prefix_sg_ids=TOKENIZER.encode(sg_prefix[idx], add_special_tokens=False)
                scenetext_ids = extract_scene_graph(
                    sg_ids=rtn_sg_ids, 
                    prefix_sg_ids=prefix_sg_ids,
                    gen_token_id=special_token_ids[prev_task], 
                    sep_token_id=special_token_ids['sep_token'], 
                    eos_token_id=special_token_ids['eos_token'],
                    max_sg_item=30,
                    model_arch=args.model_arch
                )

                q_ids, a_ids = extract_qa(
                    cqa_ids = rtn_cqa_ids,
                    que_token_id=special_token_ids[f'que_{prev_task}'], 
                    ans_token_id=special_token_ids['ans_token'], 
                    eos_token_id=special_token_ids['eos_token'],
                    model_arch=args.model_arch
                )

                n_gt_sg = np.where(np.array(prefix_sg_ids)==special_token_ids['sep_token'])[0].shape[0] + 1

                SGList.append(scenetext_ids)
                QueList.append(q_ids)
                AnsList.append(a_ids)
                N_GT_ST_List.append(n_gt_sg)

        sg_decode = TOKENIZER.batch_decode(SGList)
        que_decode = TOKENIZER.batch_decode(QueList)
        ans_decode = TOKENIZER.batch_decode(AnsList)
        
        gen_anno = []
        for idx, (sg, q, a) in enumerate(zip(sg_decode, que_decode, ans_decode)):
            ocr_token_list = extract_ocr_tokens(scene_graph_txt=sg, sep_token=special_tokens['sep_token'],ocr_token=special_tokens['ocr_token'])
            rtn_dict = collate_for_anno(
                scene_graph_txt=sg,
                que_txt=q,
                ans_txt=a, 
                ocr_token_list=ocr_token_list, 
                sep_token=special_tokens['sep_token'], 
                n_gt_sg=N_GT_ST_List[idx], 
                task_name=prev_task
            )
            gen_anno.append(rtn_dict)

        replay_save_dir = os.path.join(args.replay_dir, f"{args.model_name}_{args.cl_setting}_{order}")
        os.makedirs(replay_save_dir, exist_ok=True)
        fn = "{}_REPLAY[{}]_AT[{}].npy".format(order, prev_task[0], cur_task[0])
        np.save(os.path.join(replay_save_dir, fn), np.array(gen_anno), allow_pickle=True)
        
        _all_gen_pseudo_training_samples.extend(gen_anno)
        
        if len(gen_anno)>5000: # 5000 need to be set later
            rtn_for_train.extend(np.random.choice(gen_anno,5000,replace=False).tolist())
        else:
            rtn_for_train.extend(gen_anno)
    
    _all_gen_pseudo_training_samples = np.array(_all_gen_pseudo_training_samples)
    fn = "{}_REPLAY_AT[{}].npy".format(order, cur_task[0])
    np.save(os.path.join(replay_save_dir, fn), _all_gen_pseudo_training_samples, allow_pickle=True)
    
    model.train()
    return np.array(rtn_for_train)


def Check_Sanity(sg_ids, cqa_ids, sep_token_id, que_token_id, ans_token_id, eos_token_id, model_arch):
    np_sg_ids = np.array(sg_ids)
    np_cqa_ids = np.array(cqa_ids)
    
    if model_arch == "decoder-only":
        # check sg:
        if (que_token_id in sg_ids) or (ans_token_id in sg_ids): return False
        
        # 1. check [que], unique?
        que_indicator = np.where(np_cqa_ids==que_token_id)[0]
        if len(que_indicator)!=1: return False
        
        # 2. check [ans], unique?
        ans_indicator = np.where(np_cqa_ids==ans_token_id)[0]
        if len(ans_indicator)!=1: return False
        
        # 3. check <|endoftext|>
        eos_indicator = np.where(np_cqa_ids==eos_token_id)[0]
        if len(eos_indicator)==0: return False
        
        # 4. order: scene graph -> question -> answer? 
        que_pos = que_indicator[-1]
        ans_pos = ans_indicator[-1]
        eos_pos = eos_indicator[0]
        sep_pos = np.where(cqa_ids==sep_token_id)[0]
        last_sep_pos = 0 if len(sep_pos)==0 else sep_pos[-1]
        if not(last_sep_pos < que_pos < ans_pos < eos_pos): return False
        
        # 5. len(question), len(answer), etc
        return True
    
    else: # model_arch is "encoder-decoder"
        if eos_token_id not in sg_ids or eos_token_id not in cqa_ids:
            return False
        ctx_first_eos_pos = np.where(np_sg_ids==eos_token_id)[0][0]
        cqa_first_eos_pos = np.where(np_cqa_ids==eos_token_id)[0][0]
        np_sg_ids = np_sg_ids[:ctx_first_eos_pos+1]
        np_cqa_ids = np_cqa_ids[:cqa_first_eos_pos+1]
        
        # no [que] for encoder-decoder model
        #  check [ans], unique?
        ans_indicator = np.where(np_cqa_ids==ans_token_id)[0]
        if len(ans_indicator)!=1: return False

        if ans_token_id in sg_ids: return False

        if sep_token_id in cqa_ids: return False
        return True


def extract_scene_graph(sg_ids, prefix_sg_ids, gen_token_id, sep_token_id, eos_token_id, max_sg_item=30, model_arch="decoder-only"):
    # process and decode scene graph
    if model_arch == "decoder-only":
        np_sg_ids = np.array(sg_ids)
        gen_pos = np.where(np_sg_ids==gen_token_id)[0][0]
        sep_pos = np.where(np_sg_ids==sep_token_id)[0]
        if len(sep_pos)==0:
            last_sep_pos = -1
        else:
            last_sep_pos = sep_pos[max_sg_item] if len(sep_pos)>max_sg_item else sep_pos[-1]
        sg_text_ids = np_sg_ids[gen_pos+1:last_sep_pos].tolist()
    else: # model_arch is encoder-decoder
        np_sg_ids = np.array(sg_ids)
        ctx_first_eos_pos = np.where(np_sg_ids==eos_token_id)[0][0]
        np_sg_ids = np_sg_ids[:ctx_first_eos_pos+1]
        
        sep_pos = np.where(np_sg_ids==sep_token_id)[0]
        if len(sep_pos)==0:
            last_sep_pos = -1
        else:
            last_sep_pos = sep_pos[max_sg_item] if len(sep_pos)>max_sg_item else sep_pos[-1]
        sg_text_ids = prefix_sg_ids + [sep_token_id,] + np_sg_ids[:last_sep_pos].tolist()
    return sg_text_ids

def extract_qa(cqa_ids, que_token_id, ans_token_id, eos_token_id, model_arch="decoder-only"):
    # process and decode qa
    if model_arch == "decoder-only":
        np_cqa_ids = np.array(cqa_ids)
        que_pos = np.where(np_cqa_ids==que_token_id)[0][0]
        ans_pos = np.where(np_cqa_ids==ans_token_id)[0][0]
        eos_pos = np.where(np_cqa_ids==eos_token_id)[0][0]

        que_text_ids = np_cqa_ids[que_pos+1:ans_pos].tolist()
        ans_text_ids = np_cqa_ids[ans_pos+1:eos_pos].tolist()
    else:
        np_cqa_ids = np.array(cqa_ids)
        cqa_first_eos_pos = np.where(np_cqa_ids==eos_token_id)[0][0]
        np_cqa_ids = np_cqa_ids[:cqa_first_eos_pos+1]
        ans_pos = np.where(np_cqa_ids==ans_token_id)[0][0]

        que_text_ids = np_cqa_ids[:ans_pos]
        ans_text_ids = np_cqa_ids[ans_pos:-1]
    return que_text_ids, ans_text_ids


def model_generate_sg_batch(model, tokenizer, context, cqa_context, sep_token, gen_token, que_token, args):
    # use different method the generate scene graph given prefix of 2 sg items:
    fn_param = {
        "beam search":dict(            
            max_length = 512,
            num_beams = 5, 
            no_repeat_ngram_size=2,
            early_stopping = True
        ),
        "greedy decode":dict(max_length=512),
        "sampling":dict(
            do_sample=True, 
            max_length=512, 
            top_k=20, 
            top_p=0.9,            
        )
    }
    context_inp_dict = tokenizer.batch_encode_plus(context, padding=True, return_tensors='pt')
    for k, tensors in context_inp_dict.items():
        context_inp_dict[k] = tensors.to(model.device)
    inp_dict = fn_param["beam search"]
    inp_dict.update(context_inp_dict)
    generate_text_ids = model.generate(**inp_dict)
    generate_text_ids = generate_text_ids.cpu().numpy()

    # generate QA
    cqa_context_inp_dict = tokenizer.batch_encode_plus(cqa_context, padding=True, return_tensors='pt')
    for k, tensors in cqa_context_inp_dict.items():
        cqa_context_inp_dict[k] = tensors.to(model.device)
    inp_dict = fn_param['greedy decode']
    inp_dict.update(cqa_context_inp_dict)
    cqa_generate = model.generate(**inp_dict)
    generate_qa_ids = cqa_generate.cpu().numpy()
    
    return dict(gen_sg_ids=generate_text_ids, cqa_ids=generate_qa_ids)

def extract_ocr_tokens(scene_graph_txt, sep_token, ocr_token="[OCR]"):
    if not (ocr_token in scene_graph_txt):
        return []
    
    sg_items = scene_graph_txt.split(sep_token)
    sg_items = [sg.strip() for sg in sg_items]
    ocr_token_list = []
    for sg in sg_items:
        if ocr_token not in sg:
            continue
        # rm [OCR]
        rm_ocr_tok = (" "+sg).split(ocr_token)[-1].strip()
        ocr = rm_ocr_tok.split(" ")[0]
        ocr_token_list.append(ocr)
    return ocr_token_list


def collate_for_anno(scene_graph_txt, que_txt, ans_txt, ocr_token_list, sep_token, n_gt_sg, task_name):
    scene_graph_txt = scene_graph_txt.strip()
    que_txt = que_txt.strip()
    ans_txt = ans_txt.strip()
    
    scene_graph_list = scene_graph_txt.split(sep_token)
    scene_graph_list = [s.strip()+" {}".format(sep_token) for s in scene_graph_list]
    if n_gt_sg >= len(scene_graph_list):
        scene_graph_mask = [1,] * len(scene_graph_list)
    else:
        scene_graph_mask = [1,]*n_gt_sg + [0,]*(len(scene_graph_list)-n_gt_sg)
    
    assert len(scene_graph_list)==len(scene_graph_mask)
    
    pred_sg_list = scene_graph_list.copy()
    np.random.shuffle(pred_sg_list)

    rtn_dict = dict(
        anno_source = "{}_replay".format(args.model_name),
        stage = "{}_gen".format(task_name),
        question_id = uuid.uuid1().hex,
        question = que_txt,
        image_id = None,
        image_source = None,
        program = None,
        answer = ans_txt,
        answers = [ans_txt] * 10,
        raw_question_type = None,
        ocr = [],
        ocr_tokens = ocr_token_list,
        ocr_info=[],
        set_name = 'train',
        feature_path = None,
        supporting_fact = [],
        gqa_question = None,
        gt_scene_graph_seq = scene_graph_list,
        gt_scene_graph_mask = scene_graph_mask,
        pred_scene_graph_seq = pred_sg_list
    )
    return rtn_dict

if __name__=="__main__":
    pass