# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import warnings

import omegaconf
import torch
import numpy as np
from copy import deepcopy
from mmf.common.registry import registry
from mmf.utils.file_io import PathManager
from mmf.datasets.multi_datamodule import MultiDataModule
from mmf.modules.metrics import Metrics
from mmf.trainers.base_trainer import BaseTrainer
from mmf.trainers.callbacks.checkpoint import CheckpointCallback
from mmf.trainers.callbacks.early_stopping import EarlyStoppingCallback
from mmf.trainers.callbacks.logistics import LogisticsCallback
from mmf.trainers.callbacks.lr_scheduler import LRSchedulerCallback
from mmf.trainers.callbacks.cl_callback import CLCallback
from mmf.trainers.core.callback_hook import TrainerCallbackHookMixin
from mmf.trainers.core.device import TrainerDeviceMixin
from mmf.trainers.core.evaluation_loop import TrainerEvaluationLoopMixin
from mmf.trainers.core.profiling import TrainerProfilingMixin
from mmf.trainers.core.training_loop import TrainerTrainingLoopMixin, GEMTrainerTrainingLoopMixin
from mmf.common.CL_constant import TASK_DICT, FCL_DATA_ATTR, ABBR2TASK, get_prev_task, get_task
from mmf.utils.build import build_model, build_optimizer
from mmf.utils.general import print_model_parameters
from omegaconf import DictConfig, OmegaConf
from packaging import version
from easydict import EasyDict as edict

logger = logging.getLogger(__name__)


@registry.register_trainer("GEM")
class GEMTrainer(
    TrainerCallbackHookMixin,
    GEMTrainerTrainingLoopMixin,
    TrainerDeviceMixin,
    TrainerEvaluationLoopMixin,
    TrainerProfilingMixin,
    BaseTrainer,
):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def load(self):
        # From BaseTrainer: 
        # config run_type, print config, config device+seed+callback, load dataset + load model+ load optimizer + load metric
        super().load()
         
        self.load_fp16_scaler()

        # Callbacks
        self.on_init_start()

        # get master model before parallelization, added by wx
        self.get_model_master()

        # Parallize model
        self.parallelize_model()

        # Callbacks
        self.on_init_end()

        # init CL setting:
        self._init_CL_setting()

        # init supp dataloader
        self._init_supp_dataloader()

        # init model fields
        self._init_model_field()
        

    def _init_CL_setting(self):
        self.cl_config = self.config.training.CL
        self.task_info = edict(
            cl_setting = self.cl_config.cl_setting,
            task_order = self.cl_config.task_order,
            task_name  = self.cl_config.cur_task,
            task_abbv  = self.cl_config.cur_task[0],
            task_index = self.cl_config.task_order.index(self.cl_config.cur_task[0])
        )
    
    def _init_model_field(self):
        self.model_master.task_id = self.task_info.task_index
        if not hasattr(self.model_master, "grad_dims"):
            self.model_master.grad_dims = []
            for param in self.model_master.parameters():
                self.model_master.grad_dims.append(param.data.numel())
        if not hasattr(self.model_master, "grads"):
            self.model_master.grads = torch.zeros(sum(self.model_master.grad_dims), len(self.task_info.task_order))
            self.model_master.grads = self.model_master.grads.to(self.device)

    def get_model_master(self):
        self.model_master = self.model


    def configure_callbacks(self):
        self.checkpoint_callback = CheckpointCallback(self.config, self)
        self.early_stop_callback = EarlyStoppingCallback(self.config, self)
        self.logistics_callback = LogisticsCallback(self.config, self)
        self.lr_scheduler_callback = LRSchedulerCallback(self.config, self)
        

        # Reset callbacks as they are class variables and would be shared between
        # multiple interactive shell calls to `run`
        self.callbacks = []
        # Add callbacks for execution during events
        self.callbacks.append(self.lr_scheduler_callback)
        # checkpoint_callback needs to be called after lr_scheduler_callback so that
        # lr_scheduler_callback._scheduler.step() happens before saving checkpoints
        # (otherwise the saved last_epoch in scheduler would be wrong)

        self.callbacks.append(self.checkpoint_callback)
        self.callbacks.append(self.logistics_callback)
        # Add all customized callbacks defined by users
        for callback in self.config.training.get("callbacks", []):
            callback_type = callback.type
            callback_param = callback.params
            callback_cls = registry.get_callback_class(callback_type)
            self.callbacks.append(callback_cls(self.config, self, **callback_param))

    def load_datasets(self):
        logger.info("Loading datasets")
        self.dataset_loader = MultiDataModule(self.config)

        self.train_loader = self.dataset_loader.train_dataloader()
        self.val_loader = self.dataset_loader.val_dataloader()
        self.test_loader = self.dataset_loader.test_dataloader()

    def load_model(self):
        logger.info("Loading model")
        if self.config.model in self.config.model_config:
            attributes = self.config.model_config[self.config.model]
        else:
            warnings.warn(
                f"Model {self.config.model}'s config not present. "
                + "Continuing with empty config"
            )
            attributes = OmegaConf.create()
        # Easy way to point to config for other model
        if isinstance(attributes, str):
            attributes = self.config.model_config[attributes]

        with omegaconf.open_dict(attributes):
            attributes.model = self.config.model

        self.model = build_model(attributes)
        self.model = self.model.to(self.device)

    def load_optimizer(self):
        logger.info("Loading optimizer")
        self.optimizer = build_optimizer(self.model, self.config)

    def load_metrics(self) -> None:
        logger.info("Loading metrics")
        metrics = self.config.evaluation.get("metrics", [])
        self.metrics = Metrics(metrics)
        self.metrics_params = self.metrics.required_params

    def load_fp16_scaler(self):
        if self.training_config.fp16:
            assert version.parse(torch.__version__) >= version.parse(
                "1.6"
            ), f"Using fp16 requires torch version >- 1.6, found: {torch.__version__}"
            assert self.device != torch.device("cpu"), "fp16 cannot be used on cpu"

        set_torch_grad_scaler = True
        if self.training_config.fp16 and self.distributed:
            try:
                from fairscale.optim.grad_scaler import ShardedGradScaler
                from fairscale.optim.oss import OSS

                if isinstance(self.optimizer, OSS):
                    self.scaler = ShardedGradScaler()
                    set_torch_grad_scaler = False
                    logger.info("Using FairScale ShardedGradScaler")
            except ImportError:
                logger.info("Using Pytorch AMP GradScaler")

        if set_torch_grad_scaler:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.training_config.fp16)

    def train(self):
        logger.info("===== Model =====")
        logger.info(self.model)
        print_model_parameters(self.model)

        if "train" in self.run_type:
            self.on_train_start()
            self.training_loop()
            self.on_train_end()

        self.inference()
        self.finalize()

    def inference(self):
        dataset_type = []
        if "val" in self.run_type:
            dataset_type.append("val")
        if any(rt in self.run_type for rt in ["inference", "test", "predict"]):
            dataset_type.append("test")

        for dataset in dataset_type:
            if self.config.evaluation.predict:
                self.on_prediction_start()
                self.prediction_loop(dataset)
                self.on_prediction_end()
            else:
                self.on_test_start()
                logger.info(f"Starting inference on {dataset} set")
                report, meter = self.evaluation_loop(dataset, use_tqdm=True)
                self.on_test_end(report=report, meter=meter)

    def finalize(self):
        self.dataset_loader.teardown()
        self.teardown()

    def _init_supp_dataloader(self):
        # check sanity here
        assert self.cl_config.use_cl, "GEM must enable cl setting"
        assert self.cl_config.use_replay, "GEM must enable replaying samples"
        assert self.cl_config.replay_method == "gem", "Must use gem for replay method"

        replay_rate = self.cl_config.replay_rate
        self.supp_annos = []

        for i in range(self.task_info.task_index):
            prev_task = ABBR2TASK[self.task_info.cl_setting][self.task_info.task_order[i]]
            anno_prev = load_npy(TASK_DICT[self.task_info.cl_setting][prev_task]['train'])
            for j in range(i, self.task_info.task_index):
                n_split = j+1
                mocking_task_name = ABBR2TASK[self.task_info.cl_setting][self.task_info.task_order[j+1]]
                mocking_dta_size = FCL_DATA_ATTR[self.task_info.cl_setting][mocking_task_name]['train']['data_size']
                mocking_n_sample = np.ceil(replay_rate*mocking_dta_size/n_split).astype(np.int32)
                if mocking_n_sample < len(anno_prev):
                    anno_prev = np_set_seed_and_select(arr=anno_prev, N_select=mocking_n_sample)
            self.supp_annos.append(np.array(anno_prev))
        
        self.supp_dls = []
        for supp_anno in self.supp_annos:   # hacking for supp train loaders
            dl = deepcopy(self.train_loader)
            dl.get_datasets()[0].annotation_db = supp_anno
            dl.set_lengths()
            dl.set_samplers()
            self.supp_dls.append(dl)

def np_set_seed_and_select(arr, N_select, seed=1234):
    '''
    set seed for reproducible reply
    '''
    assert len(arr) >= N_select
    np.random.seed(seed=seed)
    rtn_arr = np.random.choice(
        arr, size=N_select, replace=False
    )
    return rtn_arr

def load_npy(path):
    with PathManager.open(path, "rb") as f:
        db = np.load(f, allow_pickle=True)
    return db