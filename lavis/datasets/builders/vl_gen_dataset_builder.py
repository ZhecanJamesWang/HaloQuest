
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.vl_gen_datasets import VL_Gen_dataset
import os
import lavis.common.utils as utils
import warnings
from lavis.common.registry import registry

@registry.register_builder("vl_gen")
class VLGenBuilder(BaseDatasetBuilder):
    train_dataset_cls = VL_Gen_dataset
    eval_dataset_cls = VL_Gen_dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask/defaults.yaml",
        "uvqa": "configs/datasets/multitask/uvqa.yaml",
    }
    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val"]:
                continue

            is_train = split == "train"
            image_size = self.config.vis_processor['train'].image_size if split=='train' else self.config.vis_processor['eval'].image_size
            if image_size<0:
                continue
            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_paths = ann_info.get(split).storage
            
            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor, text_processor=text_processor,
                ann_paths=ann_paths, image_size=image_size,split=split
            )

        return datasets

