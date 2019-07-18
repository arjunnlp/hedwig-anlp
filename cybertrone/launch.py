#!/usr/bin/env python
"""Launch training on AWS with 8 GPUs."""

from attrdict import AttrDefault
import argparse
import ncluster
import util

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='txl',
                    help="name of the current run, used for machine naming and logging directory")
parser.add_argument('--config', type=str, default='',
                    help='which training config to use')
parser.add_argument('--nospot', action='store_true',
                    help='Use more expensive on-demand instance')
parser.add_argument('--skip_setup', action='store_true',
                    help='Make startup faster by skiping various initialization tasks, like '
                         'tmux/efs setup. Only use on reruns.')
parser.add_argument('--launch_tensorboard', action='store_true',
                    help='Also launch tensorboard instance for monitoring run')

# Flags that affect all configs
parser.add_argument('--num_rings', type=int, default=16)
parser.add_argument('--image_name', type=str, default='cybertronai00',
                    help="use custom AMI ")
parser.add_argument('--conda_env', type=str, default='pytorch_p36',
                    help='use custom conda env')
args = parser.parse_args()

# Config notes:
# 'base_lr': gives learning rate relative to  BASE_LR_BATCHSIZE
# actual learning rate will multiply this by global_batch_size/BASE_LR_BATCHSIZE
# batch_size: per-GPU batch size

# "canonical" batch size os the size base lr is measured to apply linear scaling, do not change
BASE_LR_BATCHSIZE = 32

# These defaults are inherited by all configs
config_defaults = {
    'image_name': args.image_name,
    'conda_env': args.conda_env,
    'num_rings': args.num_rings,
    'architecture': 'wt103_base',
}

################################################################################
# Distributed training configs
################################################################################

# logs: yaro-1gpu.02
one_gpu = {
    'base_lr': 0.000125 * 5 / 3 * 5,
    'batch_size': 32,
    'instance_type': 'p3.2xlarge',
    'machines': 1
}

# Logs: yaro-fp16
one_machine = {
    'base_lr': 0.000125 * 5 / 3,
    'instance_type': 'p3dn.24xlarge',
    'batch_size': 96,
    'machines': 1,
}

# logs: yaro-two-fp16.04
two_machines = {
    'base_lr': 0.000125 * 5 / 3,
    'instance_type': 'p3dn.24xlarge',
    'batch_size': 96,
    'machines': 2,
}

# logs: yaro-four
four_machines = {
    'base_lr': 0.000125,
    'instance_type': 'p3dn.24xlarge',
    'batch_size': 96,
    'machines': 4,
}

# logs: /ncluster/runs/sixteen.01/info.log
sixteen_machines = {
    'base_lr': 0.000125 / 4,
    'instance_type': 'p3dn.24xlarge',
    'batch_size': 96,
    'machines': 16,
}

# logs: yaro-eight.03
eight_machines = {
    'base_lr': 0.000125 / 2,
    'instance_type': 'p3dn.24xlarge',
    'batch_size': 96,
    'machines': 8,
}



test_1 = {
    'base_lr': 0.000125 * 5 / 3,
    'instance_type': 'p3dn.24xlarge',
    'batch_size': 16,
    'architecture': 'wt103_large',
    'machines': 1,
}


test_2 = {
    'base_lr': 0.000125 * 5 / 3,
    'instance_type': 'p3dn.24xlarge',
    'batch_size': 16,
    'architecture': 'wt103_large',
    'machines': 2,
}


# logs: /ncluster/runs/four.01
test_4 = {
    'base_lr': 0.000125,
    'instance_type': 'p3dn.24xlarge',
    'batch_size': 16,
    'architecture': 'wt103_large',
    'machines': 4,
}

# logs: /ncluster/runs/eight.02/info.log
test_8 = {
    'base_lr': 0.001 / 4, # Divide by 4 to counteract batch adjustment
    'instance_type': 'p3dn.24xlarge',
    'batch_size': 16,
    'architecture': 'wt103_large',
    'machines': 8,
}

# logs: /ncluster/runs/sixteen.01/info.log
test_16 = {
    'base_lr': 0.001 / 4, # Divide by 4 to counteract batch adjustment
    'instance_type': 'p3dn.24xlarge',
    'batch_size': 16,
    'architecture': 'wt103_large',
    'machines': 16,
}

################################################################################
# Network architectures
################################################################################


# parameters from https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/run_wt103_base.sh
wt103_base = {
    'n_layer': 16,
    'd_model': 512,
    'n_head': 8,
    'd_head': 48,
    'd_inner': 2048,
    'dropout': 0.1,
    'dropatt': 0.0,
    'optim': 'lamb',
    'max_tokens': int(1.8e9),
    'tgt_len': 128,
    'mem_len': 128,
    'eval_tgt_len': 128,
}

# Parameters for train.py to match https://github.com/kimiyoung/transformer-xl/blob/master/tf/scripts/wt103_large_tpu.sh
wt103_large = {
    'n_layer': 18,
    'd_model': 1024,
    'n_head': 16,
    'd_head': 64,
    'd_inner': 4096,
    'dropout': 0.2,
    'dropatt': 0.2,
    'optim': 'lamb',
    'warmup_tokens': 0,
    'max_tokens': int(1.8e9 * 20),
    'tgt_len': 384,
    'mem_len': 384,
    'eval_tgt_len': 128,
    'fp16': True,
    'dynamic_loss_scale': True,
    'bpe': True,
    'init_std': 0.005,
    'div_val': 1,
}


def main():
    config = AttrDefault(lambda: None, config_defaults)

    assert args.config in globals(), f"unknown config {args.config}"
    config.update(eval(args.config))

    job = ncluster.make_job(name=args.name,
                            run_name=f"{args.name}",
                            num_tasks=config.machines,
                            image_name=config.image_name,
                            instance_type=config.instance_type,
                            spot=not args.nospot,
                            skip_setup=args.skip_setup)

    job.rsync('.')
    job.run(f'killall python || echo failed && '  # kill previous run
            f'source activate {config.conda_env} && ' +
            f'pip install -r requirements.txt')

    instance_info = ncluster.aws_backend.INSTANCE_INFO[config.instance_type]
    num_gpus_per_machine = instance_info['gpus']

    total_gpus = num_gpus_per_machine * config.machines
    global_batch_size = config.batch_size * total_gpus

    # linear LR scaling (https://arxiv.org/abs/1706.02677)
    lr = config.base_lr * (global_batch_size / BASE_LR_BATCHSIZE)

    # TODO(y): change dataset location to /data/transformer-xl-data after
    # image is cut
    # worker parameters with training setup
    worker_params = {
        'seed': 1111,
        'data': 'data/wikitext-103',
        'dataset': 'wt103',
        'adaptive': True,
        'log_interval': 100,
        'eval_interval': 1000,
        'logdir': job.logdir,
        'lr': lr,
        'fp16': True,
        'dynamic_loss_scale': True,
        'batch_size': config.batch_size,
    }

    if config.architecture == 'wt103_large':
        worker_params.update(wt103_large)
    elif config.architecture == 'wt103_base':
        worker_params.update(wt103_base)
    else:
        assert False, f"Uknown architecture {config.architecture}"

    nccl_params = f'NCCL_DEBUG=VERSION NCCL_MIN_NRINGS={config.num_rings} '

    for i, task in enumerate(job.tasks):
        dist_params = \
            f'--nproc_per_node={num_gpus_per_machine} ' \
            f'--nnodes={config.machines} --node_rank={i} ' \
            f'--master_addr={job.tasks[0].ip} --master_port={6016} '
        cmd = f'{nccl_params} python -m torch.distributed.launch {dist_params} ' \
            f'train.py {util.dict_to_args(worker_params)}'
        task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
        task.run(cmd, non_blocking=True)

    print(f"Logging to {job.logdir}")

    if args.launch_tensorboard:
        task = ncluster.make_task('tensorboard',
                                  instance_type='r5.large',
                                  image_name=args.image_name)

        task.run('source activate tensorflow_p36')
        task.run(f'tensorboard --logdir={ncluster.get_logdir_root()} --port=6006',
                 non_blocking=True)
        print(f'TensorBoard at http://{task.public_ip}:6006')


if __name__ == '__main__':
    main()
