import os
import json
import argparse
import logging
import datetime
logger = logging.getLogger(__name__)

import GPUtil
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIGPTConfig, T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, CONFIG_NAME
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILL_VAL = -100
LEN_FACTOR = 1.163
MEMORY_FACTOR = {
    "finetune": 0.58,
    "multitask": 0.58,
    "lll": 0.35,
    "ewc": 0.30,
    "mas": 0.18,
    "gem": 0.50,
}
TURING_ARCHS = {'Tesla V100', '2080 Ti'}
MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, 'gpt2'),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIGPTConfig,'openai-gpt'),
    't5-small': (T5ForConditionalGeneration, T5Tokenizer, T5Config, 't5-small'),
    'distilgpt2': (GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, 'distilgpt2'),
    't5v1_1-small': (T5ForConditionalGeneration, T5Tokenizer, T5Config, 'google/t5-v1_1-small')
}
SAVE_NAME = 'model-'
FINAL_SAVE_NAME = 'model-finish'

from mmf.common.CL_constant import FCL_DATA_ATTR, ABBR2TASK
fcl_data_attrs = FCL_DATA_ATTR

def ABBR2TASKList(cl_setting, abbv_seq):
    abbv_mapping = ABBR2TASK[cl_setting]
    taskList = [abbv_mapping[abbv] for abbv in abbv_seq]
    return taskList

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adam_epsilon", default=1e-4, type=float)
    parser.add_argument("--add_task_tokens", action="store_true") # use this for the first token
    parser.add_argument("--data_dir", type=str, default='/home/nus/stan/functional_continual_learning_dev/Gen_data/v0.6')
    parser.add_argument("--cl_setting", type=str, default="functional")
    parser.add_argument("--task_seq",type=str, default='oarlks')
    parser.add_argument("--train_perc",type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gen_debug", action="store_true")
    parser.add_argument("--decay_style", type=str, default="linear")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--gen_lm_sample_percentage", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=6.25e-5)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--lm_lambda", type=float, default=0.25)
    parser.add_argument("--lr_schedule", type=str, default="warmup_linear")
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--max_n_epochs", type=int, default=30)
    parser.add_argument("--min_batch_size", type=int, default=4)
    parser.add_argument("--min_n_steps", type=int, default=1500)
    parser.add_argument("--model_dir_root", type=str, default='/Users/stan/exp/QAG_debug/')
    parser.add_argument("--replay_dir", type=str, default='/Users/stan/exp/QAG_debug/replay')
    parser.add_argument("--model_name", type=str, default="distilgpt2", choices=["gpt2", "openai-gpt","t5-small", "distilgpt2", 't5v1_1-small'])
    parser.add_argument("--model_arch", type=str, default='decoder-only')
    parser.add_argument("--use_gt", action="store_true", help="whether use_gt for generation")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_train_epochs", type=int, default=15)
    parser.add_argument("--dynamic_epochs", action="store_true")
    parser.add_argument("--n_warmup_ratio", type=float, default=0.005)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--use_sep", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_train_type", type=str, default="lll", choices=["lll"])
    parser.add_argument("--tasks", nargs='+', default=["object","attribute",'relation','logical','knowledge','scenetext'])
    parser.add_argument("--skip_tasks", nargs='+')
    parser.add_argument("--test_batch_size", type=int, default=0)
    parser.add_argument("--tokens_weight", type=float, default=5)
    parser.add_argument("--train_batch_size", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--qp_margin", type=float, default=0.5)
    parser.add_argument("--n_sg_seq", type=int, default=30)
    args = parser.parse_args()

    if args.debug:
        args.logging_steps = 1
        torch.manual_seed(0)
        torch.backends.cudnn.deterministric = True

    if args.task_seq is not None:
        args.tasks = ABBR2TASKList(cl_setting=args.cl_setting, abbv_seq=args.task_seq)
        
    args.change_train_size = (args.train_perc != 1.0)
    if args.change_train_size:
        if args.model_dir_root.endswith("/"):
            args.model_dir_root = args.model_dir_root[:-1] + "_tr{}".format(args.train_perc)
        else:
            args.model_dir_root = args.model_dir_root + "_tr{}".format(args.train_perc)
    args.replay_dir = os.path.join(args.model_dir_root, f"{args.model_name}_replay")

    os.makedirs(args.model_dir_root, exist_ok=True)
    os.makedirs(args.replay_dir, exist_ok=True)
    
    args.model_dir_root = os.path.join(args.model_dir_root, args.model_name,
            args.seq_train_type, "{}_{}".format("_".join(args.tasks),
                args.gen_lm_sample_percentage) if "lll" in args.seq_train_type else "_".join(args.tasks))

    args.device_ids = GPUtil.getAvailable(maxLoad=0.5, maxMemory=0.5, limit=args.n_gpus)
    if len(args.device_ids) == 0:
        logger.error('No available GPUs!')
        raise NotImplementedError("No CPU mode available!")

    if len(args.device_ids) < args.n_gpus:
        logger.warning('Available number of GPU = {} < n_gpus = {}'.format(len(args.device_ids), args.n_gpus))
        args.n_gpus = len(args.device_ids)
        logger.warning('Continue training with {} GPUs'.format(args.n_gpus))

    torch.cuda.set_device(args.device_ids[0])

    gpus = GPUtil.getGPUs()
    gpu_names = [gpus[device_id].name for device_id in args.device_ids]
    if not all(any(turing_arch in gpu_name for turing_arch in TURING_ARCHS) for gpu_name in gpu_names):
        logger.warning('Not all gpus support fp16 training! Will use fp32 instead.')
        args.fp32 = True
    if not args.fp32:
        global MEMORY_FACTOR
        MEMORY_FACTOR = dict([k, v*1.4] for k, v in MEMORY_FACTOR.items())              # memory factor for each of the task
    args.memory_sizes = [gpus[device_id].memoryTotal for device_id in args.device_ids]  # memory size of each gpu
    args.memory_sizes[0] = args.memory_sizes[0] * (1 - 0.04 * (args.n_gpus-1))          # 
    for i in range(1, args.n_gpus):
        args.memory_sizes[i] = args.memory_sizes[i] * 1.04
    if args.train_batch_size <= 0:
        args.train_batch_size = [int(memory_size * MEMORY_FACTOR[args.seq_train_type]) for memory_size in args.memory_sizes]
    if args.test_batch_size <= 0:
        args.test_batch_size = [int(memory_size * MEMORY_FACTOR[args.seq_train_type]) for memory_size in args.memory_sizes]

    # init and config model
    special_tokens = {"question_token": "[que]", "ans_token":'[ans]',  "ocr_token":"[OCR]",}
    official_spec_tokens = {"pad_token":'[pad]', "unk_token":'[unk]', "eos_token": '<|endoftext|>', "sep_token":'[SEP]'} # add [SEP], [que] here

    # gpt, gpt2, t5
    args.model_arch = 'encoder-decoder' if "t5" in args.model_name else 'decoder-only'
    # assert args.model_arch in ['encoder-decoder', 'decoder-only']
    
    model_class, tokenizer_class, config_class, pretrained_pth = MODEL_CLASSES[args.model_name]
    args.load_model_name = pretrained_pth

    tokenizer = tokenizer_class.from_pretrained(pretrained_pth)
    tokenizer.add_special_tokens(official_spec_tokens)
    tokenizer.add_tokens(list(special_tokens.values()))
    special_tokens.update(official_spec_tokens)
    special_token_ids = {k:tokenizer.convert_tokens_to_ids(v) for k,v in special_tokens.items()}

    model_config = config_class.from_pretrained(pretrained_pth)
    model_config.vocab_size = len(tokenizer)
    tokens_weight = torch.ones([model_config.vocab_size], dtype=torch.float).cuda()
    tokens_weight[special_token_ids["ans_token"]] = args.tokens_weight
    tokens_weight[special_token_ids["question_token"]] = args.tokens_weight
    tokens_weight[special_token_ids["ocr_token"]] = args.tokens_weight
    tokens_weight[special_token_ids["sep_token"]] = args.tokens_weight
    
    tokenizer.padding_side = "left"
    args.max_len = getattr(model_config, 'n_positions', 512)

    data_attrs = fcl_data_attrs

    if args.seq_train_type == "multitask":
        args.n_train_epochs = {'_'.join(args.tasks): args.n_train_epochs}

    else:
        if args.dynamic_epochs:
            data_sizes = {task: data_attrs[args.cl_setting][task]["train"]["data_size"] for task in args.tasks}
            max_total_data_size = max(data_sizes.values()) * args.n_train_epochs
            args.n_train_epochs = {d[0]: min(args.max_n_epochs, max_total_data_size//d[1]) for d in data_sizes.items()}
        else:
            args.n_train_epochs = {task: args.n_train_epochs for task in args.tasks}

    return args, model_config, model_class, tokenizer, config_class, special_token_ids, special_tokens, data_attrs, tokens_weight


class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
          last = self.last
        except AttributeError:
          last = record.relativeCreated

        delta = record.relativeCreated/1000 - last/1000
        record.relative = "{:.1f}".format(delta)
        record.uptime = str(datetime.timedelta(seconds=record.relativeCreated//1000))
        self.last = record.relativeCreated
        return True


def init_logging(filename):
    logging_format = "%(asctime)s - %(uptime)s - %(relative)ss - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(format=logging_format, filename=filename, filemode='a', level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(logging_format))
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    for handler in root_logger.handlers:
        handler.addFilter(TimeFilter())

args, MODEL_CONFIG, MODEL_CLASS, TOKENIZER, CONFIG_CLASS, SPECIAL_TOKEN_IDS, SPECIAL_TOKENS, DATA_ATTRS, TOKENS_WEIGHT = parse_args()

from mmf.common.CL_constant import GENERATED_SG_PTH as mmf_gen_sg_pth
GENERATED_SG_PTH = mmf_gen_sg_pth

from mmf.common.CL_constant import DATA_DIR as mmf_data_dir
DATA_DIR = mmf_data_dir

from mmf.common.CL_constant import TASK_DICT as mmf_task_dict
TASK_DICT = mmf_task_dict
for cl_setting in TASK_DICT:
    for stage in TASK_DICT[cl_setting]:
        TASK_DICT[cl_setting][stage].update({"n_train_epochs": args.n_train_epochs})
