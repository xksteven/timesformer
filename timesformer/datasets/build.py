# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from fvcore.common.registry import Registry
from .vce_dataloader import VCEDataset
from .v2v_dataloader import V2VPairwiseDataset, V2VListwiseDataset

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""


def build_dataset(dataset_name, cfg, split, listwise=False):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the dataset to be constructed.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    # Capitalize the the first letter of the dataset_name since the dataset_name
    # in configs may be in lowercase but the name of dataset class should always
    # start with an uppercase letter.
    if dataset_name == "emotions_vce":
        return VCEDataset(cfg.DATA.PATH_TO_DATA_DIR, split=split, video_width=256, video_height=256, num_frames=8)
    elif dataset_name == "emotions_v2v":
        if listwise:
            return V2VListwiseDataset(cfg.DATA.PATH_TO_DATA_DIR, split=split, video_width=256, video_height=256, num_frames=8)
        elif split == "train":
            return V2VPairwiseDataset(cfg.DATA.PATH_TO_DATA_DIR, split=split, video_width=256, video_height=256, num_frames=8)
        elif split == "test":
            return V2VPairwiseDataset(cfg.DATA.PATH_TO_DATA_DIR, split=split, video_width=256, video_height=256, num_frames=8)
    name = dataset_name.capitalize()
    return DATASET_REGISTRY.get(name)(cfg, split)
