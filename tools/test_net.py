# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
from fvcore.common.file_io import PathManager
import cv2
from einops import rearrange, reduce, repeat
import scipy.io

import timesformer.utils.checkpoint as cu
import timesformer.utils.distributed as du
import timesformer.utils.logging as logging
import timesformer.utils.misc as misc
import timesformer.visualization.tensorboard_vis as tb
from timesformer.datasets import loader
from timesformer.models import build_model
from timesformer.utils.meters import TestMeter
from tqdm import tqdm


logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_v2v_test(test_loader, listwise_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    pairwise_error = 0

    test_meter.iter_tic()
    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].float().cuda(non_blocking=True)
            else:
                inputs = inputs.float().cuda(non_blocking=True)

        test_meter.data_toc()

        # logits = model(inputs)
        logits_0 = model(inputs[0])
        logits_1 = model(inputs[1])

        diffs = (logits_1 - logits_0).squeeze(dim=1)

        preds = diffs
        labels = torch.ones(diffs.shape[0])

        top1_err = None

        # # Compute the errors.
        errors = torch.sigmoid(preds) > 0.5 
        top1_err = torch.sum(1.0 - errors.float()) / diffs.size(0) 

        if cfg.NUM_GPUS > 1:
            top1_err = du.all_reduce([top1_err])


        # Copy the errors from GPU to CPU (sync point).
        top1_err = top1_err[0].item()
        pairwise_error += top1_err

        test_meter.iter_toc()

        # Update and log stats.
        test_meter.update_stats(
            preds.cpu().detach(), labels.detach(), video_idx.detach()
        )
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()


    logger.info(
            f"Pairwise error = {pairwise_error / len(test_loader)}"
        )


    # Listwise evaluation
    all_predictions = []
    for cur_iter, (inputs, _, _, meta) in enumerate(tqdm(listwise_loader)):
        #logger.info(f"len inputs = {len(inputs)}\nsize of inputs0 = {inputs[0].size()} ")
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].float().cuda(non_blocking=True)
            else:
                inputs = inputs.float().cuda(non_blocking=True)

        #logger.info(f"listwise meta = {meta}")
        # lets just run the whole set through the model in one go.
        stacked_inputs = torch.cat(inputs, 0)
        #logger.info(f"stacked_inputs size = {stacked_inputs.size()}")
        logits = model(stacked_inputs)
       
        if cfg.NUM_GPUS > 1:
            preds = du.all_gather(logits)

        tmp = torch.stack(preds, 0).T

        preds = tmp
 
        for pred in preds:
            all_predictions.append(pred.cpu().data.numpy())

    logger.info(all_predictions)
    correct = 0
    for sublist in tqdm(all_predictions):
       # predict everything in order
       prev = -float('inf')
       curr_correct = True
       # Correct predictions must be in sorted order (small to large)
       for val in sublist:
           if val < prev:
               curr_correct = False
               break
           else:
               prev = val
       if curr_correct:
           correct += 1

       # Correct if the first video is rated best.
       # if np.argmax(sublist) == 0:
       #     correct += 1

    total = len(all_predictions)
    logger.warn("V2V List wise Total: {}, Correct: {} Accuracy: {}\n".format(total, correct, correct / total))

    print("V2V List wise Total: {}, Correct: {} Accuracy: {}\n".format(total, correct, correct / total))

    logger.info(f"len listwise = {len(listwise_loader)}, gpus * len loader = {cfg.NUM_GPUS * len(listwise_loader)}\n")

    return test_meter


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True).float()
            else:
                inputs = inputs.cuda(non_blocking=True).float()

            # Transfer the data to the current GPU device.
            if isinstance(labels, (list,)):
                new_labels = labels[0]
                new_labels = new_labels.unsqueeze(1)
                for i in range(1, len(labels)):
                    tmp_labels = labels[i].unsqueeze(1)
                    new_labels = torch.cat((new_labels, tmp_labels), dim=1)

                #logger.info(f"new_labels size = {new_labels.size()}")
                labels = new_labels
            labels = labels.float().cuda(non_blocking=True)
            video_idx = video_idx.cuda(non_blocking=True)
            #for key, val in meta.items():
            #    if isinstance(val, (list,)):
            #        for i in range(len(val)):
            #            val[i] = val[i].cuda(non_blocking=True)
            #    else:
            #        meta[key] = val.cuda(non_blocking=True)

        test_meter.data_toc()

        # Perform the forward pass.
        preds = model(inputs)

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather(
                [preds, labels, video_idx]
            )
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()

        if len(labels.size()) > 1:
            one_hot_labels = torch.argmax(labels, dim=1)
        else:
            one_hot_labels = labels
 
        test_meter.iter_toc()

        #logger.warn(f"preds = {preds.size()} labels = {labels.size()} video_idx = {video_idx.size()}")

        # Update and log stats.
        test_meter.update_stats(
            preds.detach(), one_hot_labels.detach(), video_idx.detach()
        )
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    all_preds = test_meter.video_preds.clone().detach()
    all_labels = test_meter.video_labels
    if cfg.NUM_GPUS:
        all_preds = all_preds.cpu()
        all_labels = all_labels.cpu()
    if writer is not None:
        writer.plot_eval(preds=all_preds, labels=all_labels)

    if cfg.TEST.SAVE_RESULTS_PATH != "":
        save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

        with PathManager.open(save_path, "wb") as f:
            pickle.dump([all_labels, all_labels], f)

        logger.info(
            "Successfully saved prediction results to {}".format(save_path)
        )

    test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    if "v2v" in cfg.TEST.DATASET:
        test_loader = loader.construct_loader(cfg, "test")
        listwise_loader = loader.construct_loader(cfg, "test", listwise=True)
    else:
        test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (
        len(test_loader.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    # Create meters for multi-view testing.
    test_meter = TestMeter(
        len(test_loader.dataset)
        // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
    )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    if "v2v" in cfg.TEST.DATASET:
        test_meter = perform_v2v_test(test_loader, listwise_loader, model, test_meter, cfg, writer)
    else:
        test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()
