from asyncio import gather
import os
import abc
import logging
import torch
import numpy as np
import pickle as pkl
from sklearn.cluster import KMeans
from copy import deepcopy
from torch.optim import Optimizer, SGD
from mmf.common.registry import registry
from mmf.common.CL_constant import get_task, get_prev_task
from mmf.trainers.callbacks.base import Callback
from mmf.utils.configuration import get_mmf_env
from mmf.common.sample import to_device
from mmf.utils.general import clip_gradients, extract_loss, get_max_updates
from mmf.utils.distributed import byte_tensor_to_object, object_to_byte_tensor
from mmf.common.CL_constant import FCL_DATA_ATTR
from mmf.common.report import Report
from easydict import EasyDict as edict
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Regularizer(abc.ABC):
    def __init__(self, config, model, dataloaders, task, prev_task=None):
        self.config = config
        self.model = None
        self.parallel_model = None
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            self.model = model.module
            self.parallel_model = model
        else: # w/o parallel
            self.model = model
            self.parallel_model = model
        self.dataloaders = deepcopy(dataloaders)
        self.task = task
        self.prev_task = prev_task
        
    @abc.abstractmethod
    def task_start_do(self):
        return NotImplemented
    
    @abc.abstractmethod
    def task_end_do(self):
        return NotImplemented
    
    def save_reg_params(self):
        save_dir = get_mmf_env(key="save_dir")
        reg_params_path = os.path.join(save_dir, "models", "reg_params.pkl")
        with open(reg_params_path, 'wb') as f:
            pkl.dump(
                dict(reg_params = self.model.reg_params, name2reg_params=self.model.name2reg_params), f
            )
        
            
    def load_reg_params(self):
        do_resume = (self.prev_task is not None) or (self.config.training.CL.reg_params_pth is not None)
        if not do_resume:
            return
        
        # higher priority for specifying resume path for debugging
        if self.config.training.CL.reg_params_pth:
            assert self.prev_task is not None, f"There is no prev_task for current task {self.task} for order {self.config.training.CL.task_order}."
            with open(self.config.training.CL.reg_params_pth, 'rb') as f:
                dict_reg = pkl.load(f)
                self.model.reg_params = dict_reg['reg_params']
                self.model.name2reg_params = dict_reg['name2reg_params']
            return

        if self.prev_task:
            save_dir = get_mmf_env(key='save_dir')
            resume_dir = save_dir.replace(self.task, self.prev_task)
            reg_params_path = os.path.join(resume_dir, "models", "reg_params.pkl")
            with open(reg_params_path, 'rb') as f:
                dict_reg = pkl.load(f)
                self.model.reg_params = dict_reg['reg_params']
                self.model.name2reg_params = dict_reg['name2reg_params']
            return


class MAS(Regularizer):
    def task_start_do(self,freeze_layers=[]):
        self.load_reg_params()
        task_start_do(self.config, self.model, freeze_layers)
    def task_end_do(self):
        updater = Omega_update(self.config, self.model.parameters(), lr=0.0001, momentum=0.9)
        compute_importance(self.config, self.model, self.parallel_model, updater, self.dataloaders)
        accumulate_reg_params(self.config, self.model)
        self.save_reg_params()

class EWC(Regularizer):
    def task_start_do(self,freeze_layers=[]):
        self.load_reg_params()
        task_start_do(self.config, self.model, freeze_layers)
    def task_end_do(self):
        updater = Omega_update(self.config, self.model.parameters(), lr=0.0001, momentum=0.9)
        compute_importance(self.config, self.model, self.parallel_model, updater, self.dataloaders, loss_type="ewc")
        accumulate_reg_params(self.config, self.model)
        self.save_reg_params()

def task_start_do(config, model, freeze_layers=[]):
    if not hasattr(model,"reg_params"):
        initialize_reg_params(config, model, freeze_layers)
    else: # load from pth
        clean_omega_sum(config, model, freeze_layers)

def initialize_reg_params(config, model, freeze_layers=[]):
    """initialize an omega for each parameter to zero"""
    reg_params={}
    name2reg_params={}
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            # print('initializing param',name)
            omega = torch.FloatTensor(param.size()).zero_()
            omega = omega.to(config.device)
            init_val = param.data.clone()
            init_val = init_val.to(config.device)
            reg_param = {}
            reg_param['omega'] = omega
            reg_param['omega_sum'] = omega
            #initialize the initial value to that before starting training
            reg_param['init_val'] = init_val
            reg_params[param] = reg_param
            name2reg_params[name] = param

    if 'data_count' not in reg_params:
        reg_params['data_count'] = 0
    reg_params['lambda'] = config.training.CL.reg_lambda
    model.reg_params = reg_params
    model.name2reg_params = name2reg_params


def clean_omega_sum(config, model,freeze_layers=[]):
    # fix loading issue:
    reg_params = {}
    name2reg_params={}
    for name, param in model.named_parameters():
        if name in model.name2reg_params:
            save_param = model.name2reg_params[name]
            assert torch.all(save_param == param)
            reg_params[param] = deepcopy(model.reg_params[save_param])
            name2reg_params[name] = param
    reg_params['lambda'] = config.training.CL.reg_lambda
    model.reg_params = reg_params
    model.name2reg_params = name2reg_params
    
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            omega = torch.FloatTensor(param.size()).zero_()
            omega = omega.to(config.device)
            reg_param = model.reg_params.get(param)
            reg_param['omega_sum'] = omega
            model.reg_params[param] = reg_param
    model.reg_params['data_count'] = 0

# omega of task1 + omega of task2 ...
# new_omega=omega_sum/data_count; omega=new_omega+prev_omega
def accumulate_reg_params(config, model, freeze_layers=[]):
    """accumelate the newly computed omega with the previously stroed one from the old previous tasks"""
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            if param in model.reg_params:
                reg_param=model.reg_params.get(param)
                # print('restoring previous omega',name)
                prev_omega=reg_param.get('omega')
                new_omega=reg_param.get('omega_sum') / model.reg_params["data_count"]
                acc_omega=torch.add(prev_omega, new_omega)

                del reg_param['omega_sum']
                reg_param['omega'] = acc_omega

                model.reg_params[param]=reg_param
                del prev_omega
                del new_omega
                del acc_omega
        else:
            if param in model.reg_params:
                reg_param = model.reg_params.get(param)
                # print('removing unused omega', name)
                del reg_param['omega']
                del model.reg_params[param]


# update omega for one task; use in compute_importance
class Omega_update(SGD):
    """
    Update the paramerter importance using the gradient of the function output norm. To be used at deployment time.
    reg_params:parameters omega to be updated
    batch_index,batch_size:used to keep a running average over the seen samples
    """
    def __init__(self, config, params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):

        super(Omega_update, self).__init__(params,lr,momentum,dampening,weight_decay,nesterov)
        self.config = config

    def __setstate__(self, state):
        super(Omega_update, self).__setstate__(state)

    def step(self, reg_params, batch_size, closure=None):
        """
        Performs a single parameters importance update setp
        """
        #print('************************DOING A STEP************************')
        reg_params['data_count'] += batch_size
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            #if the parameter has an omega to be updated
            for p in group['params']:

                #print('************************ONE PARAM************************')

                if p.grad is None:
                    continue

                if p in reg_params:
                    #HERE MAS IMPOERANCE UPDATE GOES
                    #get the gradient
                    unreg_dp = p.grad.data.clone()
                    reg_param = reg_params.get(p)
                    #get parameter omega
                    omega = reg_param.get('omega_sum')
                    if self.config.training.CL.reg_type == "ewc":
                        omega = omega.add((unreg_dp)**2)
                    else:
                        omega = omega.add(unreg_dp.abs_())
                    reg_param['omega_sum'] = omega
                    reg_params[p] = reg_param
                    #HERE MAS IMPOERANCE UPDATE ENDS

        return loss#HAS NOTHING TO DO

# update omega for one task
def compute_importance(config, model, parallel_model, updater, dataloaders, loss_type="l2"):
    """Mimic the depoloyment setup where the model is applied on some samples and those are used to update the importance params
       Uses the L2norm of the function output. This is what we MAS uses as default
    """
    # model.eval()  # Set model to training mode so we get the gradient
    # train_loss_fct = DataParallelCriterion(CrossEntropyLoss(ignore_index=FILL_VAL), args.device_ids)

    softmax = torch.nn.Softmax(dim=-1)
    if loss_type == "l2":
        loss_fct = torch.nn.MSELoss(reduction='mean')
    elif loss_type == "l1":
        loss_fct = torch.nn.L1Loss(reduction='mean')
    elif loss_type == "ewc":
        assert config.training.CL.reg_type == 'ewc', "loss type is ewc, pls set reg_type to be ewc"
        loss_fct = torch.nn.Identity()
    
    # parallel loss
    if registry.get("data_parallel", False):
        loss_fct = torch.nn.DataParallel(loss_fct)
    elif registry.get("distributed", False):
        loss_fct = torch.nn.parallel.DistributedDataParallel(
                    loss_fct,
                    device_ids=[config.trainer.local_rank],
                    output_device=config.trainer.local_rank,
                    find_unused_parameters=config.training.find_unused_parameters,
                )
    else:
        loss_fct = loss_fct


    for _, batch in tqdm(enumerate(dataloaders), total=len(dataloaders)):
        updater.zero_grad()
        prepared_batch = to_device(batch, config.device)
        with torch.cuda.amp.autocast(enabled=config.training.fp16):
            model_output = parallel_model(prepared_batch)
            report = Report(prepared_batch, model_output)
        
        loss = None
        if config.training.CL.reg_type == 'ewc':
            loss = extract_loss(report, loss_divisor=1) # set update_freq to be 1 here.

        else: #not 'ewc' type loss
            logits = report['scores']
            logits = softmax(logits)
            target_zeros = torch.zeros(logits.size()).to(config.device)
            loss = loss_fct(logits, target_zeros).mean()
        
        report.detach()
        config.trainer.scaler.scale(loss).backward()
        config.trainer.scaler.step(updater, model.reg_params, report.batch_size)
        config.trainer.scaler.update()
        

REG_DICT = {
    "ewc": EWC, 
    "mas": MAS
}

def init_reg(config, trainer):
    reg_type = config.training.CL.reg_type
    if not reg_type in REG_DICT:
        raise NotImplementedError
    reg_class = REG_DICT[reg_type]

    # config, model, dataloaders, task, prev_task=None
    cur_task = config.training.CL.cur_task # TODO
    prev_task = get_prev_task(task_abbv_order = config.training.CL.task_order, cur_task_abbv=config.training.CL.cur_task[0], cl_setting=config.training.CL.cl_setting)[1] # prev_task name
    reg_instance = reg_class(
        config = config, 
        model = trainer.model,
        dataloaders = trainer.train_loader, 
        task = cur_task, 
        prev_task = prev_task
    )
    return reg_instance


# ==================================== call backs =======================================
# for ewc/mas
@registry.register_callback("clcallback")
class CLCallback(Callback):
    def __init__(self, config, trainer):
        super().__init__(config, trainer)
        # what we have from super class:
        # self.trainer = trainer
        # self.config = config
        # self.training_config = self.config.training
        self.call_back_type = "continual_learning_call_back"
        self.config = edict(config)
        self.config.trainer = self.trainer
        self.config.device = self.trainer.device
        self.reg_type = self.config.training.CL.reg_type

        self.regularizer = init_reg(self.config, self.trainer)

    def on_train_start(self, **kwargs): # TODO
        logger.info(f" ==== CL Regularizer {self.reg_type} task start === ")
        self.regularizer.task_start_do()

    def on_train_end(self, **kwargs): # TODO
        self.regularizer.task_end_do()
        logger.info(f"===== CL Regularizer {self.reg_type} task end ====")


@registry.register_callback("kmeans_store")
class KmeansStore(Callback):
    def __init__(self, config, trainer):
        super().__init__(config, trainer)
        self.call_back_type = "kmeans_store_call_back"
        self.cl_config = registry.get("config").training.CL
        self.valid_check()
        self.model = self.trainer.model
        logger.info("======= Will use kmeans replay ========")
        
    def valid_check(self):
        # check if it is legal to use Kmeans_store callback
        # (0) is continual learning
        assert self.cl_config.use_cl, "using kmeans store call back should enable CL setting."
        # (1) use kmeans
        assert self.cl_config.use_replay and self.cl_config.replay_method == "kmeans", "using kmeans store call back should enable kmeans replay method."
        assert isinstance(self.cl_config.replay_rate, float), "kmeans must use valid replay rate"
        # (2) use cls
        model_cfg = getattr(self.config.model_config,'unicl') if hasattr(self.config.model_config, 'unicl') else getattr(self.config.model_config, 'unicl_debug')
        assert model_cfg.use_cls, "using kmeans store call back should enable using cls embedding for model."

    def on_train_end(self, **kwargs):
        # gather info and cls representation
        gather_info = self._gather_info()

        # return prev set & cur set
        prev_qids, cur_info = self._divide_prev_cur(gather_info)

        qid_to_save = []
        # random select previous q into buffer
        task_index, task_name = get_task(task_abbv_order = self.cl_config.task_order, task_abbv=self.cl_config.cur_task[0], cl_setting=self.cl_config.cl_setting)
        
        tot_buffer_size = int(FCL_DATA_ATTR[self.cl_config.cl_setting][task_name]["train"]['data_size'] * self.cl_config.replay_rate)
        tot_split = task_index + 1

        k_buffer_cur = int(tot_buffer_size / tot_split)
        k_buffer_prev = tot_buffer_size - k_buffer_cur

        if k_buffer_prev==0: assert len(prev_qids)==0
        else:
            if k_buffer_prev >= len(prev_qids):
                qid_to_save.extend(prev_qids)
            else:
                qid_to_save.extend(np.random.choice(prev_qids, k_buffer_prev, replace=False).tolist())

        # kmeans and select those closet to clusters into buffer
        kmeans = KMeans(n_clusters=k_buffer_cur, verbose=True)
        X = np.array(cur_info.cls_repr)
        qids = np.array(cur_info.question_ids, dtype=object)
        kmeans.fit(X)

        clu_points_dist = kmeans.transform(X)
        select_indices = clu_points_dist.argmin(axis=0)
        select_indices = np.unique(select_indices)
        qid_to_save.extend(qids[select_indices].tolist())
        
        # store into path for later replay
        qid_to_save = set(qid_to_save)
        annos_to_save = []

        for item in self.trainer.train_loader.get_datasets()[0].annotation_db:
            if item['question_id'] in qid_to_save:
                annos_to_save.append(item)
        
        save_dir = get_mmf_env(key="save_dir")
        save_path = os.path.join(save_dir, "kmeans_replay.npy")
        np.save(save_path, np.array(annos_to_save))

    def _gather_info(self):
        gather_info = edict(
            cls_repr = [],
            question_ids = [],
            stages = []
        )
        logger.info("============== kmeans store: Gathering info ===============")
        dta_loader = self.trainer.train_loader
        with torch.no_grad():
            self.model.eval()
            for _, batch in tqdm(enumerate(dta_loader), total=len(dta_loader)):
                prepared_batch = to_device(batch, self.trainer.device)
                with torch.cuda.amp.autocast(enabled=self.config.training.fp16):
                    model_output = self.model(prepared_batch)
                report = Report(prepared_batch, model_output)
                report.detach()

                cls_outp, question_id, stage = report.cls_output.cpu(), report.question_id.cpu(), report.stage.cpu()
                cls_outp = cls_outp.squeeze(1)

                for c_, qid_, stage_ in zip(cls_outp, question_id, stage):
                    gather_info.cls_repr.append(c_.numpy())
                    gather_info.question_ids.append(byte_tensor_to_object(qid_))
                    gather_info.stages.append(byte_tensor_to_object(stage_))

        return gather_info
    

    def _divide_prev_cur(self, info):
        prev_qids = []
        cur_info = edict(
            cls_repr = [],
            question_ids = []
        )

        for c_, qid_, s_ in zip(info.cls_repr, info.question_ids, info.stages):
            if "replay" in s_:
                prev_qids.append(qid_)
            else:
                cur_info.cls_repr.append(c_)
                cur_info.question_ids.append(qid_)
        
        return prev_qids, cur_info
