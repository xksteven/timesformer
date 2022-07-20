# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# A script with a list of commands for submitting SLURM jobs

#SBATCH --job-name=timesformer
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=xksteven@gmail.com

## %j is the job id, %u is the user id
#SBATCH --output=/path/to/output/logs/slog-%A-%a.out

## filename for job standard error output (stderr)
#SBATCH --error=/data/hendrycks/timesformer/slurm_outputs/slog-%A-%a.err

#SBATCH --array=1
#SBATCH --partition=partition_of_your_choice
#SBATCH --nodes=1 -C volta32gb
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=480GB
#SBATCH --signal=USR1@600
#SBATCH --time=72:00:00
#SBATCH --open-mode=append

conda activate timesformer

WORKINGDIR=/data/hendrycks/timesformer
CURPYTHON=python

srun --label ${CURPYTHON} ${WORKINGDIR}/tools/run_net.py --cfg ${WORKINGDIR}/configs/emotions/TimeSformer_divST_8x32_224_4gpus.yaml NUM_GPUS 8 TRAIN.BATCH_SIZE 8

# python tools/run_net.py \
#   --cfg /data/hendrycks/timesformerconfigs/emotions/TimeSformer_divST_8x32_224_4gpus.yaml
