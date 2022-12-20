# Copyright (c) Facebook, Inc. and its affiliates.
import os
import logging
import random
import collections
import numpy as np
import torch
from mmf.common.registry import registry
from mmf.utils.file_io import PathManager
from mmf.common.sample import Sample
from mmf.common.CL_constant import ABBR2TASK, DATA_DIR, TASK_DICT, FCL_DATA_ATTR, get_task
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.distributed import byte_tensor_to_object, object_to_byte_tensor
from mmf.utils.text import word_tokenize
from mmf.datasets.base_dataset import BaseDataset
from mmf.datasets.databases.annotation_database import AnnotationDatabase
from mmf.datasets.databases.features_database import FeaturesDatabase
from mmf.datasets.databases.image_database import ImageDatabase
from easydict import EasyDict as edict
logger = logging.getLogger(__name__)

class CLVQADataset(BaseDataset):
    def __init__(self, config, dataset_type="train", index=0, *args, **kwargs):
        super().__init__("clvqa", config, dataset_type, *args, **kwargs)
        self.cl_config = registry.get("config").training.CL
        self.dataset_type = dataset_type
        self._default_index = index # this is required by mmf_dataset_builder, can fix this later
        
        self.task_list = self._get_task_list(self.cl_config.task_order)
        self.task_name = self.cl_config.cur_task
        
        # to create dummy img feature for replay samples
        self.n_img_feat = self.config.n_img_feat
        self.img_feat_dim = self.config.img_feat_dim
        
        # annotation_DB
        self.build_annotation_db()
        self.supp_annotation_db()
        
        # image_DB
        self._use_images = self.config.get("use_images", False)
        if self._use_images:
            self.build_image_db()        
        # feature DB
        self._use_features = self.config.get("use_features", False)
        if self._use_features:
            self.build_features_db()
        
        self.use_ocr = self.config.use_ocr
        self.use_ocr_info = self.config.use_ocr_info
        self.mask_img_prob = self.config.mask_img_prob
        self.mask_img = (self.config.use_mask_img and dataset_type=="train")
        self.use_gt_sg = self.config.use_gt_sg
   
    def build_annotation_db(self):
        anno_paths = self._sanity_check_rtn_paths('annotations')
        if anno_paths is not None:
            paths = [self._get_path_based_on_index(self.config, "annotations", i) for i in range(len(anno_paths))]
            dbs = [self._load_npy(path) for path in paths]
            self.annotation_db = np.concatenate(dbs)

        else:
            raise ValueError("Got None for anno_paths.")

    def supp_annotation_db(self):
        if self.dataset_type == "train" and self.cl_config.use_cl and self.cl_config.use_replay:
            replay_method = self.cl_config.replay_method
            if replay_method == 'random': # replay real sample
                logger.info("======= Random Replay with real samples =======")
                replay_rate = self.cl_config.replay_rate
                supp_anno = self._rnd_replay_sample_for_cur_stage(prob=replay_rate)    
                self.annotation_db = np.concatenate([self.annotation_db, supp_anno])
            elif replay_method == "kmeans": # replay real sample
                task_index = self.task_list.index(self.task_name)
                if task_index>0:
                    logger.info("============ Random replay with kmeans ==============")
                    replay_dir = self.cl_config.replay_dir
                    replay_pth = os.path.join(replay_dir, "kmeans_replay.npy")
                    replayList = self._load_npy(replay_pth)
                    for item in replayList:
                        if "replay" not in item["stage"]:
                            item["stage"] = item["stage"] + "_replay"
                    self.annotation_db = np.concatenate([self.annotation_db, replayList])
                else:
                    logger.info("============= Random replay with kmeans, no replay at stage 1 =================")
            elif replay_method=='restore_with_number': # restore with QAG
                logger.info("======= Restore generated samples with Numbers =======")
                restore_dir = self.cl_config.restore_dir
                max_restore_per_stage = self.cl_config.max_restore_per_stage
                restoreList = []
                if self.cl_config.restore_paths is not None:
                    for path in self.cl_config.restore_paths.split(","):
                        restore_anno = self._load_npy(os.path.join(restore_dir, path))
                        if len(restore_anno) > max_restore_per_stage:
                            restore_anno = np.random.choice(restore_anno, max_restore_per_stage, replace=False)
                        restoreList.append(restore_anno)
                    restoreList = np.concatenate(restoreList)
                    self.annotation_db = np.concatenate([self.annotation_db, restoreList])
            elif replay_method == "restore_with_prob":  # restore with QAG
                logger.info("======= Restore generated samples with ratio param =======")
                restore_rate = self.cl_config.restore_rate
                restoreList = self._restore_generated_sample_for_cur_stage(ratio=restore_rate)
                self.annotation_db = np.concatenate([self.annotation_db, restoreList])
            elif replay_method == "gem":
                return
            else:
                raise NotImplementedError("Not implement replay method {}.".format(replay_method))      
    
    def build_image_db(self):
        # keep as mmf dataset
        image_path = self._get_path_based_on_index(self.config, "images", self._default_index)
        return ImageDatabase(self.config, image_path, annotation_db=self.annotation_db)
    
    def build_features_db(self):
        feat_paths = self._sanity_check_rtn_paths('features')
        if feat_paths is not None:
            paths = [self._get_path_based_on_index(self.config, "features", i) for i in range(len(feat_paths))]
            feat_dbs = [FeaturesDatabase(config=self.config, path=path, annotation_db=self.annotation_db) for path in paths]
            
            gqa_feat_db = None
            textvqa_feat_db = None
            # init db of gqa and text_vqa
            for (p,db) in zip(paths, feat_dbs):
                if "gqa" in p:
                    assert ("ocr" not in p) and ("textvqa" not in p)
                    gqa_feat_db = db
                elif "textvqa" in p or "ocr" in p:
                    assert "gqa" not in p
                    textvqa_feat_db = db
            self.features_db = edict(gqa_feat_db = gqa_feat_db, textvqa_feat_db = textvqa_feat_db)
        else:
            raise ValueError("Got None for feat_paths")


    def preprocess_sample_info(self, sample_info):
        # path = self._get_path_based_on_index(self.config, "annotations", self._index)
        # NOTE, TODO: Code duplication w.r.t to STVQA, revisit
        # during dataset refactor to support variable dataset classes
        if sample_info['image_source']=='textvqa':
            feature_path = sample_info['feature_path']
            if not feature_path.startswith('train'):
                append = 'train'
                feature_path = append + '/' + feature_path
            sample_info['feature_path'] = feature_path
            # pass
        return sample_info

    def postprocess_evalai_entry(self, entry):
        return entry  # Do nothing

    def format_for_prediction(self, report):
        answer_processor = self.answer_processor

        batch_size = len(report.question_id)
        pred_answers = report.scores.argmax(dim=-1).view(batch_size, -1)
        answer_space_size = answer_processor.get_true_vocab_size()

        image_ids = report.image_id.cpu().numpy()
        context_tokens = report.context_tokens.cpu().numpy()
        predictions = []
        for idx, question_id in enumerate(report.question_id):
            # collect VQA answers
            image_id = byte_tensor_to_object(image_ids[idx])
            tokens = byte_tensor_to_object(context_tokens[idx])
            answer_words = []
            pred_source = []
            for answer_id in pred_answers[idx].tolist():
                if answer_id >= answer_space_size:
                    answer_id -= answer_space_size
                    answer_words.append(word_tokenize(tokens[answer_id]))
                    pred_source.append("OCR")
                else:
                    if answer_id == answer_processor.EOS_IDX:
                        break
                    answer_words.append(
                        answer_processor.answer_vocab.idx2word(answer_id)
                    )
                    pred_source.append("VOCAB")
            # join all the answer tokens with space
            # (this should be correct for almost all cases)
            pred_answer = " ".join(answer_words).replace(" 's", "'s")
            entry = {
                "question_id": question_id.item(),
                "image_id": image_id,
                "answer": pred_answer,
                "pred_source": pred_source,
            }
            entry = self.postprocess_evalai_entry(entry)

            predictions.append(entry)

        return predictions

    def __len__(self):
        return len(self.annotation_db)
    
    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]                   # read from annotation with ext: .npy
        sample_info = self.preprocess_sample_info(sample_info)
        current_sample = Sample()
        
        stage = sample_info['stage']
        mask_img_feat = True if ("_gen" in stage) or ("_##" in stage) else False # hack: (1)_gen in stage means
        current_sample.stage = object_to_byte_tensor(stage)
        # iamge_id
        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = str(sample_info["image_id"])
        else:
            current_sample.image_id = sample_info["image_id"]
        # question_id
        current_sample.question_id = object_to_byte_tensor(sample_info['question_id'])
        
        if self._use_features and not mask_img_feat:
            features = self._read_features(sample_info)
            current_sample.update(features)
        elif mask_img_feat:
            current_sample.image_feature_0 = torch.zeros((self.n_img_feat, self.img_feat_dim))
            current_sample.image_info_0 = Sample(dict(max_features=torch.tensor(self.n_img_feat)))
        
        current_sample = self.add_sample_details(sample_info, current_sample)
        current_sample = self.add_answer_info(sample_info, current_sample)
        
        # only the 'max_features' key is needed
        # pop other keys to minimize data loading overhead
        if hasattr(current_sample, "image_info_0"):
            for k in list(current_sample.image_info_0):
                if k != "max_features":
                    current_sample.image_info_0.pop(k)
        if hasattr(current_sample, "image_info_1"):
            for k in list(current_sample.image_info_1):
                if k != "max_features":
                    current_sample.image_info_1.pop(k)
        else:
            current_sample.image_info_1 = current_sample.image_info_0.copy()
        return current_sample

    def add_sample_details(self, sample_info, sample):
        sample.image_id = object_to_byte_tensor(sample.image_id) # object_to_byte_tensor(sample_info['image_id']) -> tensor([  0,  26, 128,  ...,   0,   0,   0], dtype=torch.uint8)

        # 1. Load text (question words)
        question_str = (
            sample_info["question"]
            if "question" in sample_info
            else sample_info["question_str"]
        )
        text_processor_args = {"text": question_str}

        if "question_tokens" in sample_info:
            text_processor_args["tokens"] = sample_info["question_tokens"]

        processed_question = self.text_processor(text_processor_args)

        if "input_ids" in processed_question:
            sample.text = processed_question["input_ids"]             # input_id with padding
            sample.text_len = torch.tensor(                           # input tokens, w/o padding
                len(processed_question["tokens"]), dtype=torch.long
            )
        else:
            # For GLoVe based processors
            sample.text = processed_question["text"]
            sample.text_len = processed_question["length"]

        # 2. Load object
        ## added: use bbox in feature file
        if hasattr(sample.image_info_0, 'bbox'):  
            orig_boxes = sample.image_info_0.bbox
            w, h = sample.image_info_0.image_width, sample.image_info_0.image_height
            normalized_boxes = orig_boxes / np.array([w,h,w,h]).astype(np.float32)
            sample.obj_bbox_coordinates = self.copy_processor(
                {"blob": normalized_boxes}
            )["blob"]
        else:
            sample.obj_bbox_coordinates = torch.zeros((self.n_img_feat, 4))
        ## end -  added
        
        # 3. Load  SceneGraph
        scene_graph_seq = None
        if ("replay" in sample_info["stage"]) and self.use_gt_sg:
            scene_graph_seq =  sample_info["gt_scene_graph_seq"]
        else:
            scene_graph_seq = sample_info['pred_scene_graph_seq']
        scene_graph_seq = " ".join(scene_graph_seq)
        if scene_graph_seq.endswith(" [SEP]"): # remove the last [SEP] since processor will handle this
            scene_graph_seq = scene_graph_seq[:-(len(" [SEP]"))]
        sg_text_processor_args = {'text': scene_graph_seq}
        processed_sg_str = self.scene_graph_processor(sg_text_processor_args)
        if "input_ids" in processed_sg_str:
            sample.sg_text = processed_sg_str['input_ids']
            sample.sg_text_len = torch.tensor(
                len(processed_sg_str['tokens']), dtype=torch.long
            )
        else: # for GLoVe processors
            sample.sg_text = processed_sg_str["text"]
            sample.sg_text_len = processed_sg_str["length"]
        
        # 4. Load knowledge -supporting facts
        supp_fact = sample_info['supporting_fact']
        if len(supp_fact) == 0: # other stages than knowledge   
            knowledge_str = []
        
        else:    ############ processing knowledge in the field `triplet` ##########
            knowledge_str = " ".join(supp_fact[0]['triplet'])
        
        knowledge_txt_processor_args = {'text': knowledge_str}
        processed_knowledge_str = self.text_processor(knowledge_txt_processor_args)
        if "input_ids" in processed_knowledge_str:
            sample.knowledge_text = processed_knowledge_str['input_ids']
            sample.knowledge_text_len = torch.tensor(
                len(processed_knowledge_str['tokens']), dtype=torch.long
            )
        else: # for GLoVe processors
            sample.knowledge_text = processed_knowledge_str["text"]
            sample.knowledge_text_len = processed_knowledge_str["length"]
        
        # 5. Load OCR
        if not self.use_ocr:
            # remove all OCRs from the sample
            # (i.e. make an empty OCR list)
            sample_info["ocr_tokens"] = []
            sample_info["ocr_info"] = []
            if "ocr_normalized_boxes" in sample_info:
                sample_info["ocr_normalized_boxes"] = np.zeros((0, 4), np.float32)
            # clear OCR visual features
            if "image_feature_1" in sample:
                sample.image_feature_1 = torch.zeros_like(sample.image_feature_1)
            # added by lei: dummy ocr frcn feature:
            else:
                sample.image_feature_1 = torch.zeros_like(sample.image_feature_0)
            return sample

        # added: handle dataset not in textvqa
        if not hasattr(sample, "image_feature_1"):
            sample.image_feature_1 = torch.zeros_like(sample.image_feature_0)
        
        # added: mask image feature with mask_img_prob, use just Q+sg -> ans
        if self.mask_img and random.random() <= self.mask_img_prob:
            sample.image_feature_0 = torch.zeros_like(sample.image_feature_0)

        # Preprocess OCR tokens
        if hasattr(self, "ocr_token_processor"):
            ocr_tokens = [
                self.ocr_token_processor({"text": token})["text"]
                for token in sample_info["ocr_tokens"]
            ]
        else:
            ocr_tokens = sample_info["ocr_tokens"]
        
        # Get FastText embeddings for OCR tokens
        context = self.context_processor({"tokens": ocr_tokens})
        sample.context = context["text"]                        # tensor: (max_len, dim) (50, 300) here
        sample.ocr_tokens = context["tokens"]                   # tokens with padding tokens: ['aaa','bbb','ccc', ... , <pad>,<pad>,...] max_len:50

        sample.context_tokens = object_to_byte_tensor(context["tokens"])     # convert to tensor Size([4094], torch.uint8)
        sample.context_feature_0 = context["text"]                           # tensor: (max_len, dim) (50, 300) here
        sample.context_info_0 = Sample()
        sample.context_info_0.max_features = context["length"]               # tensor: len w/o padding

        # Get PHOC embeddings for OCR tokens
        if hasattr(self, "phoc_processor"):
            context_phoc = self.phoc_processor({"tokens": ocr_tokens})  
            sample.context_feature_1 = context_phoc["text"]                 # tensor: (max_len, dim) (50, 604) here
            sample.context_info_1 = Sample()
            sample.context_info_1.max_features = context_phoc["length"]     # tensor: len w/o padding

        # OCR order vectors
        if self.config.get("use_order_vectors", False):
            order_vectors = np.eye(len(sample.ocr_tokens), dtype=np.float32)    # init: len w/ padding tokens
            order_vectors = torch.from_numpy(order_vectors)
            order_vectors[context["length"] : ] = 0
            sample.order_vectors = order_vectors

        # OCR bounding box information
        if "ocr_normalized_boxes" in sample_info and hasattr(self, "copy_processor"):
            # New imdb format: OCR bounding boxes are already pre-computed
            max_len = self.config.processors.answer_processor.params.max_length     # fetch from config: 50 here
            sample.ocr_bbox_coordinates = self.copy_processor(                      # copy_processor yields a [100,4] tensor (padding with 0-vectors to length 100), fetch top max_len
                {"blob": sample_info["ocr_normalized_boxes"]}
            )["blob"][:max_len]
        elif self.use_ocr_info and "ocr_info" in sample_info:
            # Old imdb format: OCR bounding boxes are computed on-the-fly
            # from ocr_info
            sample.ocr_bbox_coordinates = self.bbox_processor(                      # yields the normalized ocr bbox as in the above if branch
                {"info": sample_info["ocr_info"]}
            )["bbox"].coordinates

        return sample

    def add_answer_info(self, sample_info, sample):
        # Load real answers from sample_info
        answers = sample_info.get("answers", [])
        answer_processor_arg = {"answers": answers}

        answer_processor_arg["tokens"] = sample.pop("ocr_tokens", [])

        processed_answers = self.answer_processor(answer_processor_arg)

        assert not self.config.fast_read, (
            "In CLDataset, online OCR sampling is incompatible "
            "with fast_read, so fast_read is currently not supported."
        )

        sample.update(processed_answers)
        sample.answers = object_to_byte_tensor(answers)

        if "answers_scores" in sample:
            sample.targets = sample.pop("answers_scores")

        return sample

    def _read_features(self, sample_info):
        if sample_info['image_source'] == 'textvqa':
            return self.features_db.textvqa_feat_db.get(sample_info)
        else:
            return self.features_db.gqa_feat_db.get(sample_info)
    
    def _sanity_check_rtn_paths(self, attribute):
        attr_config = self.config.get(attribute, None)
        if (
            self.dataset_type not in attr_config  # e.g. in config, dataset.feature.train
            or len(attr_config.get(self.dataset_type, [])) == 0
        ):
            raise ValueError(f"No {attribute} present for type {self.dataset_type}")
        
        paths = attr_config[self.dataset_type]
        return paths
    
    def _get_path_based_on_index(self, config, attribute, index):
        if attribute not in config:                                 # arg: attribute: "annotations", "iamges", "features"
            raise ValueError(f"{attribute} not present in config")

        config = config.get(attribute, None) # e.g. in config, dataset.feature

        if (
            self.dataset_type not in config  # e.g. in config, dataset.feature.train
            or len(config.get(self.dataset_type, [])) == 0
        ):
            raise ValueError(f"No {attribute} present for type {self.dataset_type}")

        paths = config[self.dataset_type]

        if isinstance(paths, str):
            selected_path = paths
        else:
            assert isinstance(paths, collections.abc.MutableSequence)
            selected_path = paths[index]

        selected_path = self._add_root_dir(selected_path)

        return selected_path
    
    
    def _add_root_dir(self, path):
        path = path.split(",")                                  # xxx-detectron.imdb,yyy-detectron.imdb
        for idx, p in enumerate(path):
            path[idx] = os.path.join(self.config.data_dir, p)

        return ",".join(path)
    
    def _load_npy(self, path):
        with PathManager.open(path, "rb") as f:
            db = np.load(f, allow_pickle=True)
        return db

    def _rnd_replay_sample_for_cur_stage(self, prob=0.2):
        # call this func after init annotation_db
        cl_setting = self.cl_config.cl_setting
        supp_anno = []
        cur_stage_idx = self.task_list.index(self.task_name)
        for i, prev_task in enumerate(self.task_list):
            if i == cur_stage_idx:
                # add mask flag in the field "stage"
                if self.cl_config.replay_mask_img:
                    logger.info("===== Mask img for random replay =====")
                    for item in supp_anno:
                        item['stage'] = item['stage'] + "_replay_##"           # hack: use "_##" to denote mask img replay
                else:
                    for item in supp_anno:
                        item['stage'] = item['stage'] + "_replay"
                return np.array(supp_anno)
            
            # else: handle previous stages:
            anno_prev = self._load_npy(TASK_DICT[cl_setting][prev_task][self.dataset_type])
            for j in range(i, cur_stage_idx):
                n_split = j + 1
                mocking_dta_size = FCL_DATA_ATTR[cl_setting][self.task_list[j+1]][self.dataset_type]['data_size']
                mocking_n_sample = np.ceil(prob * mocking_dta_size / n_split).astype(np.int32)
                if mocking_n_sample < len(anno_prev):
                    anno_prev = np_set_seed_and_select(
                        arr=anno_prev, N_select=mocking_n_sample
                    )
            supp_anno.extend(anno_prev.tolist())
    
    def _restore_generated_sample_for_cur_stage(self, ratio=0.2):
        cl_setting = self.cl_config.cl_setting
        restore_dir = self.cl_config.restore_dir
        restore_paths = self.cl_config.restore_paths.split(",")
        cur_stage_idx = self.task_list.index(self.task_name)
        assert len(restore_paths)==cur_stage_idx, f"Expected {cur_stage_idx} flies, but got {len(restore_paths)}."
        restore_list = []
        for i, prev_task in enumerate(self.task_list):
            if i == cur_stage_idx:
                return np.array(restore_list)
            
            # else: handle previous stages
            generated_prev = self._load_npy(os.path.join(restore_dir, restore_paths[i]))
            for j in range(i, cur_stage_idx):
                n_split = j+1
                mocking_dta_size = FCL_DATA_ATTR[cl_setting][self.task_list[j+1]][self.dataset_type]['data_size']
                mocking_n_sample = np.ceil(ratio * mocking_dta_size / n_split).astype(np.int32)
                if mocking_n_sample < len(generated_prev):
                    generated_prev = np_set_seed_and_select(
                        arr = generated_prev, N_select=mocking_n_sample
                    )
            restore_list.extend(generated_prev.tolist())
             
    
    def _get_task_list(self, abbv_task_order):
        if not isinstance(abbv_task_order, str):
            raise ValueError("Task order must be a string for abbv of tasks. e.g: oarlks")
        cl_setting = self.cl_config.cl_setting
        rtn_list = [ABBR2TASK[cl_setting][abbv_t] for abbv_t in abbv_task_order]
        return rtn_list
        
        
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
