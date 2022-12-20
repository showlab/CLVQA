import os

version = 1.0
STAGES_NAME = {
    "functional": dict(
        stages_name = ['object', 'attribute', 'relation', 'logical', 'knowledge', 'scenetext'],
        abbv2stage = dict(o='object', a='attribute', r='relation', l='logical', k='knowledge', s='scenetext')
    ),
    "scene": dict(
        stages_name = ["a#ShopAndDining", "b#Workplace", "c#HomeOrHotel", "d#Transportation", "e#SportAndLeisure", "f#Outdoors"],
        abbv2stage = dict(a="a#ShopAndDining", b="b#Workplace", c="c#HomeOrHotel", d="d#Transportation", e="e#SportAndLeisure", f="f#Outdoors")
    ),
}

ABBR2TASK = dict(
    functional = dict (o='object', a='attribute', r='relation', l='logical', k='knowledge', s='scenetext'),
    scene = dict(a='a#ShopAndDining', b='b#Workplace', c='c#HomeOrHotel', d="d#Transportation",e='e#SportAndLeisure', f='f#Outdoors'),
)

DATA_DIR = dict(        # modify path
    functional = "/Users/stan/code/functional_continual_learning_dev/Gen_data/v0.6",
    scene = "/Users/stan/code/functional_continual_learning_dev/Gen_data/m_scene",
)

N_TESTING_SAMPLES = dict(
    functional = dict(o=3000, a=3000, r=3000, l=3000, k=3000, s=3000),
    scene = dict(a=3000, b=3000, c=3000, d=3000, e=3000, f=3000, g=3000),
)

TASK_DICT = dict(
    functional = {
            "object":{
                "train": os.path.join(DATA_DIR["functional"],"fcl_mmf_object_train.npy"),
                "val": os.path.join(DATA_DIR["functional"],"fcl_mmf_object_val.npy"),
                "test": os.path.join(DATA_DIR["functional"],"fcl_mmf_object_val.npy"),
            },
            "attribute":{
                "train": os.path.join(DATA_DIR["functional"],"fcl_mmf_attribute_train.npy"),
                "val": os.path.join(DATA_DIR["functional"],"fcl_mmf_attribute_val.npy"),
                "test": os.path.join(DATA_DIR["functional"],"fcl_mmf_attribute_val.npy"),
            },
            "relation":{
                "train": os.path.join(DATA_DIR["functional"],"fcl_mmf_relation_train.npy"),
                "val": os.path.join(DATA_DIR["functional"],"fcl_mmf_relation_val.npy"),
                "test": os.path.join(DATA_DIR["functional"],"fcl_mmf_relation_val.npy"),
            },
            "logical":{
                "train": os.path.join(DATA_DIR["functional"],"fcl_mmf_logical_train.npy"),
                "val": os.path.join(DATA_DIR["functional"],"fcl_mmf_logical_val.npy"),
                "test": os.path.join(DATA_DIR["functional"],"fcl_mmf_logical_val.npy"),
            },
            "knowledge":{
                "train": os.path.join(DATA_DIR["functional"],"fcl_mmf_knowledge_train.npy"),
                "val": os.path.join(DATA_DIR["functional"],"fcl_mmf_knowledge_val.npy"),
                "test": os.path.join(DATA_DIR["functional"],"fcl_mmf_knowledge_val.npy"),
            },
            "scenetext":{
                "train": os.path.join(DATA_DIR["functional"],"fcl_mmf_scenetext_train.npy"),
                "val": os.path.join(DATA_DIR["functional"],"fcl_mmf_scenetext_val.npy"),
                "test": os.path.join(DATA_DIR["functional"],"fcl_mmf_scenetext_val.npy"),
            }
    },
    scene = {
        "a#ShopAndDining":{
            "train": os.path.join(DATA_DIR["scene"], "fcl_mmf_a#ShopAndDining_train.npy"),
            "val": os.path.join(DATA_DIR["scene"], "fcl_mmf_a#ShopAndDining_val.npy"),
            "test": os.path.join(DATA_DIR["scene"], "fcl_mmf_a#ShopAndDining_val.npy")
        },
        "b#Workplace":{
            "train": os.path.join(DATA_DIR["scene"], "fcl_mmf_b#Workplace_train.npy"),
            "val": os.path.join(DATA_DIR["scene"], "fcl_mmf_b#Workplace_val.npy"),
            "test": os.path.join(DATA_DIR["scene"], "fcl_mmf_b#Workplace_val.npy")
        },
        "c#HomeOrHotel":{
            "train": os.path.join(DATA_DIR["scene"], "fcl_mmf_c#HomeOrHotel_train.npy"),
            "val": os.path.join(DATA_DIR["scene"], "fcl_mmf_c#HomeOrHotel_val.npy"),
            "test": os.path.join(DATA_DIR["scene"], "fcl_mmf_c#HomeOrHotel_val.npy"),
        },
        "d#Transportation":{
            "train": os.path.join(DATA_DIR["scene"], "fcl_mmf_d#Transportation_train.npy"),
            "val": os.path.join(DATA_DIR["scene"], "fcl_mmf_d#Transportation_val.npy"),
            "test": os.path.join(DATA_DIR["scene"], "fcl_mmf_d#Transportation_val.npy"),
        },
        "e#SportAndLeisure":{
            "train": os.path.join(DATA_DIR["scene"], "fcl_mmf_e#SportAndLeisure_train.npy"),
            "val": os.path.join(DATA_DIR["scene"], "fcl_mmf_e#SportAndLeisure_val.npy"),
            "test": os.path.join(DATA_DIR["scene"], "fcl_mmf_e#SportAndLeisure_val.npy"),
        },
        "f#Outdoors":{
            "train": os.path.join(DATA_DIR["scene"], "fcl_mmf_f#Outdoors_train.npy"),
            "val": os.path.join(DATA_DIR["scene"], "fcl_mmf_f#Outdoors_val.npy"),
            "test": os.path.join(DATA_DIR["scene"], "fcl_mmf_f#Outdoors_val.npy"),
        },
    },

)

FCL_DATA_ATTR = dict(
    functional = {
        "object": {
            "train":{"data_size":20000},
            "val":{"data_size":3000},
            "test":{"data_size":3000},    
        },
        "attribute": {
            "train":{"data_size":18673},
            "val":{"data_size":3000},
            "test":{"data_size":3000},    
        },
        "relation": {
            "train":{"data_size":20000},
            "val":{"data_size":3000},
            "test":{"data_size":3000},    
        },
        "logical":{
            "train":{"data_size":20000},
            "val":{"data_size":3000},
            "test":{"data_size":3000},    
        },
        "knowledge":{
            "train":{"data_size":20000},
            "val":{"data_size":3000},
            "test":{"data_size":3000},    
        },
        "scenetext": {
            "train":{"data_size":16868},
            "val":{"data_size":2422},
            "test":{"data_size":2422},
        }
    },
    scene = {
        "a#ShopAndDining":{
            "train":{"data_size":20000},
            "val":{"data_size":3000},
            "test":{"data_size":3000},   
        },
        "b#Workplace": {
            "train":{"data_size":20000},
            "val":{"data_size":3000},
            "test":{"data_size":3000},
        },
        "c#HomeOrHotel":{
            "train":{"data_size":20000},
            "val":{"data_size":3000},
            "test":{"data_size":3000},         
        },
        "d#Transportation":{
            "train":{"data_size":20000},
            "val":{"data_size":3000},
            "test":{"data_size":3000},               
        },
        "e#SportAndLeisure":{
            "train":{"data_size":20000},
            "val":{"data_size":3000},
            "test":{"data_size":3000},            
        },
        "f#Outdoors":{
            "train":{"data_size":20000},
            "val":{"data_size":3000},
            "test":{"data_size":3000},              
        },
    },
 
)

GENERATED_SG_PTH = dict(
    functional = "/Users/stan/code/functional_continual_learning_dev/SG_processing/generated_sg_all_stages_v6.json", # modify path here
    scene = "/Users/stan/code/functional_continual_learning_dev/SG_processing/stage_sg_scene_setting_50u-50c.json",  # modify path here
)


def get_task(task_abbv_order, task_abbv, cl_setting='functional'):
    info = ABBR2TASK[cl_setting]
    task_name = info[task_abbv]
    task_index = task_abbv_order.index(task_abbv)
    return task_index, task_name

def get_prev_task(task_abbv_order, cur_task_abbv, cl_setting='functional'):
    info = ABBR2TASK[cl_setting]
    cur_task_index = task_abbv_order.index(cur_task_abbv)
    if cur_task_index == 0:
        return None, None
    else:
        prev_task_abbv = task_abbv_order[cur_task_index-1]
        prev_task = info[prev_task_abbv]
        prev_task_index = cur_task_index - 1
        return prev_task_index, prev_task
