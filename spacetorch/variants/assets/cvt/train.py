from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pprint
import time

import torch
import torch.nn.parallel
import torch.optim
from torch.utils.collect_env import get_pretty_env_info
from tensorboardX import SummaryWriter

from spacetorch.variants.assets.cvt.config import config
from spacetorch.variants.assets.cvt.config import update_config
from spacetorch.variants.assets.cvt.config import save_config
from spacetorch.variants.assets.cvt.core.loss import build_criterion
from spacetorch.variants.assets.cvt.core.function import train_one_epoch, test
from spacetorch.variants.assets.cvt.dataset import build_dataloader
from spacetorch.variants.assets.cvt.models import build_model
from spacetorch.variants.assets.cvt.optim import build_optimizer
from spacetorch.variants.assets.cvt.scheduler import build_lr_scheduler
from spacetorch.variants.assets.cvt.utils.comm import comm
from spacetorch.variants.assets.cvt.utils.utils import create_logger
from spacetorch.variants.assets.cvt.utils.utils import init_distributed
from spacetorch.variants.assets.cvt.utils.utils import setup_cudnn
from spacetorch.variants.assets.cvt.utils.utils import summary_model_on_master
from spacetorch.variants.assets.cvt.utils.utils import resume_checkpoint
from spacetorch.variants.assets.cvt.utils.utils import save_checkpoint_on_master
from spacetorch.variants.assets.cvt.utils.utils import save_model_on_master


def main(args, model_wrapper, spatial_args, positions, layernames):
    init_distributed(args)
    setup_cudnn(config)

    update_config(config, args)
    final_output_dir = create_logger(config, args.cfg, 'train')
    tb_log_dir = final_output_dir

    # Make sure logs are saved to file
    log_file = os.path.join(final_output_dir, "train.log")
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)

    logging.info(f"Logging to {log_file}")

    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> using {} GPUs".format(args.num_gpus))

        output_config_path = os.path.join(final_output_dir, 'config.yaml')
        logging.info("=> saving config into: {}".format(output_config_path))
        save_config(config, output_config_path)

    model = model_wrapper(build_model(config), spatial_args, positions, layernames)
    model.to(torch.device('cuda'))

    # copy model file
    summary_model_on_master(model, config, final_output_dir, True)

    if config.AMP.ENABLED and config.AMP.MEMORY_FORMAT == 'nhwc':
        logging.info('=> convert memory format to nhwc')
        model.to(memory_format=torch.channels_last)

    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    best_perf = 0.0
    best_model = True
    begin_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = build_optimizer(config, model)

    best_perf, begin_epoch = resume_checkpoint(
        model, optimizer, config, final_output_dir, True
    )

    train_loader = build_dataloader(config, True, args.distributed)
    valid_loader = build_dataloader(config, False, args.distributed)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    criterion = build_criterion(config)
    criterion.cuda()
    criterion_eval = build_criterion(config, train=False)
    criterion_eval.cuda()

    lr_scheduler = build_lr_scheduler(config, optimizer, begin_epoch)

    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP.ENABLED)

    logging.info('=> start training')
    for epoch in range(begin_epoch, config.TRAIN.END_EPOCH):
        head = 'Epoch[{}]:'.format(epoch)
        logging.info('=> {} epoch start'.format(head))

        start = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        logging.info('=> {} train start'.format(head))
        with torch.autograd.set_detect_anomaly(config.TRAIN.DETECT_ANOMALY):
            train_one_epoch(config, train_loader, model, criterion, optimizer,
                            epoch, final_output_dir, tb_log_dir, writer_dict,
                            scaler=scaler)
        logging.info(
            '=> {} train end, duration: {:.2f}s'
            .format(head, time.time()-start)
        )

        # evaluate on validation set
        logging.info('=> {} validate start'.format(head))
        val_start = time.time()

        if epoch >= config.TRAIN.EVAL_BEGIN_EPOCH:
            perf = test(
                config, valid_loader, model, criterion_eval,
                final_output_dir, tb_log_dir, writer_dict,
                args.distributed
            )

            best_model = (perf > best_perf)
            best_perf = perf if best_model else best_perf

        logging.info(
            '=> {} validate end, duration: {:.2f}s'
            .format(head, time.time()-val_start)
        )

        lr_scheduler.step(epoch=epoch+1)

        if (epoch % 25 == 0 or epoch == config.TRAIN.END_EPOCH - 1) and comm.is_main_process():
            save_model_on_master(
                model, args.distributed, final_output_dir, f'model_{epoch}.pth'
            )
        
        save_checkpoint_on_master(
            model=model,
            distributed=args.distributed,
            model_name=config.MODEL.NAME,
            optimizer=optimizer,
            output_dir=final_output_dir,
            in_epoch=True,
            epoch_or_step=epoch,
            best_perf=best_perf,
        )

        if best_model and comm.is_main_process():
            save_model_on_master(
                model, args.distributed, final_output_dir, 'model_best.pth'
            )

        logging.info(
            '=> {} epoch end, duration : {:.2f}s'
            .format(head, time.time()-start)
        )

    save_model_on_master(
        model, args.distributed, final_output_dir, 'final_state.pth'
    )

    if config.SWA.ENABLED and comm.is_main_process():
        save_model_on_master(
             args.distributed, final_output_dir, 'swa_state.pth'
        )

    writer_dict['writer'].close()
    logging.info('=> finish training')


if __name__ == '__main__':
    main()
