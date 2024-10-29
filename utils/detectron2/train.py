import os
import copy
import torch
import detectron2
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, build_detection_train_loader
import detectron2.data.transforms as T

setup_logger()

path = '/data/ephemeral/home'

# register dataset
# with k-fold dataset (refer to kfold_split.py)
num = 1
register_coco_instances('coco_trash_train', {}, f'{path}/dataset/split/train_42_fold_{num}.json', f'{path}/dataset')
register_coco_instances('coco_trash_val', {}, f'{path}/dataset/split/val_42_fold_{num}.json', f'{path}/dataset')

MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal",
                                                         "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

# configuration
def setup_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))

    cfg.DATASETS.TRAIN = ('coco_trash_train',)
    cfg.DATASETS.TEST = ('coco_trash_val',)

    cfg.DATALOADER.NUM_WOREKRS = 2

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/retinanet_R_101_FPN_3x.yaml')

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 15000
    cfg.SOLVER.STEPS = (8000,12000)
    cfg.SOLVER.GAMMA = 0.005
    cfg.SOLVER.CHECKPOINT_PERIOD = 3000

    cfg.OUTPUT_DIR = f'{path}/output_retinanet'

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    cfg.MODEL.RETINANET.NUM_CLASSES = 10  # RetinaNet용 설정

    cfg.TEST.EVAL_PERIOD = 3000

    return cfg

# data augmentation
def MyMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')

    transform_list = [
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomCrop('absolute', (256, 256)),
        T.RandomRotation([-30,30]),
        T.RandomLighting(scale=0.1)
    ]

    image, transforms = T.apply_transform_gens(transform_list, image)

    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]

    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)

    return dataset_dict

class MyTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
            cfg, mapper=MyMapper, sampler=sampler
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(f'{path}/output_eval', exist_ok=True)
            output_folder = f'{path}/output_eval'

        return COCOEvaluator(dataset_name, cfg, False, output_folder)
    
# configuration setting
cfg = setup_config()

# Train
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# validation
evaluator = COCOEvaluator('coco_trash_val', cfg, False, output_dir=f'{path}/validation')
val_loader = build_detection_test_loader(cfg, 'coco_trash_val'
inference_on_dataset(trainer.model, val_loader, evaluator)