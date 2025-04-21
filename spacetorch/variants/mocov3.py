import os
import argparse
import json
import sys
import math
import torch
import wandb
import shutil
import builtins
import shutil
import importlib.util
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path
from functools import partial
from torch.utils.tensorboard import SummaryWriter

import spacetorch.variants.assets.moco.loader as loader
import spacetorch.variants.assets.moco.builder as builder
from spacetorch.utils.torch_utils import resolve_sequential_module_from_str
from spacetorch.utils.generic_utils import convert_to_serializable
from spacetorch.variants.base_arch import BaseArch
from spacetorch.variants.positions import NetworkPositions
from spacetorch.losses.losses_torch import spatial_correlation_loss
from spacetorch.utils.gpu_utils import is_master_process
from spacetorch.variants.assets.moco.vits import vit_base
from spacetorch.variants.assets.moco.main_moco import train
from spacetorch.variants.assets.moco.main_lincls import main_worker as eval_main_worker


class MoCov3(BaseArch):
    """
    Implements the MoCov3 self-supervised objective function.
    """
    def __init__(self):
        super().__init__()
    
    def get_model_layernames(self, args):
        return list(args.spatial_loss.spatial_weight.keys())

    def set_cfg(self, args):
        self.cfg = argparse.Namespace(**{
            "spatial_args": args,
            "save_dir": Path(args.output_dir) / "eval",
            "data": args.variant.params.dataset_path,
            "arch": "vit_base",
            "epochs": 300,
            "workers": 32,
            "start_epoch": 0,
            "batch_size": 1024,
            "lr": 1.5e-4,
            "momentum": 0.9,
            "weight_decay": .1,
            "print_freq": 10,
            "resume": "",
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "dist_url": "tcp://localhost:10001",
            "dist_backend": "nccl",
            "seed": None,
            "gpu": None,
            "multiprocessing_distributed": True,
            "moco_dim": 256,
            "moco_mlp_dim": 4096,
            "moco_m": 0.99,
            "moco_m_cos": True,
            "moco_t": .2,
            "stop_grad_conv1": True,
            "optimizer": "adamw",
            "warmup_epochs": 40,
            "crop_min": 0.08,
        })

    def set_eval_cfg(self, args):
        self.eval_cfg = argparse.Namespace(**{
            "spatial_args": args,
            "save_dir": Path(args.output_dir) / "eval",
            "data": args.variant.params.dataset_path,
            "arch": "vit_base",
            "epochs": 30,
            "workers": 32,
            "start_epoch": 0,
            "batch_size": 1024,
            "lr": 3,
            "momentum": 0.9,
            "weight_decay": 0.,
            "print_freq": 10,
            "resume": "",
            "evaluate": False,
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "dist_url": "tcp://localhost:10001",
            "dist_backend": "nccl",
            "seed": None,
            "gpu": None,
            "multiprocessing_distributed": True,
            "pretrained": args.variant.params.pretrained_weights,
        })

    def _load_pretrained_weights(self, args, model):
        checkpoint = torch.load(args.variant.params.pretrained_weights)
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.model.base_encoder') and not k.startswith('module.model.base_encoder.%s' % "head"):
                # remove prefix
                state_dict[k[len("module.model.base_encoder."):]] = state_dict[k]
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % "head"):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        print(("Pretrained weights found at {} and loaded with msg: {}".format(args.variant.params.pretrained_weights, msg)))

    def _load_positions(self, position_dir):
        network_positions = NetworkPositions.load_from_dir(position_dir)
        network_positions.to_torch()
        return network_positions.layer_positions

    def set_model(self, args):
        sys.path.insert(0, str(Path(args.variant.setup.model).parent))
        self.layernames = self.get_model_layernames(args)
        self.positions = self._load_positions(args.spatial_loss.position_dir)
        pass

    def set_eval_model(self, args):
        vits = load_function_from_file(args.variant.setup.eval_model, "vits")
        self.eval_model = vits.vit_base()
        if args.variant.params.pretrained_weights:
            self._load_pretrained_weights(args, self.eval_model)
        self.eval_model.cuda()

    def start_training_protocol(self, args):
        if self.cfg.dist_url == "env://" and self.cfg.world_size == -1:
            self.cfg.world_size = int(os.environ["WORLD_SIZE"])

        self.cfg.distributed = self.cfg.world_size > 1 or self.cfg.multiprocessing_distributed

        ngpus_per_node = torch.cuda.device_count()
        if self.cfg.multiprocessing_distributed:
            self.cfg.world_size = ngpus_per_node * self.cfg.world_size
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, self.cfg, self.positions, self.layernames))
        else:
            main_worker(self.cfg.gpu, ngpus_per_node, self.cfg, self.positions, self.layernames)

    def set_eval_protocol(self, args):
        self.eval_main = eval_main_worker

    def start_eval_protocol(self):
        if self.eval_cfg.dist_url == "env://" and self.eval_cfg.world_size == -1:
            self.eval_cfg.world_size = int(os.environ["WORLD_SIZE"])

        self.eval_cfg.distributed = self.eval_cfg.world_size > 1 or self.eval_cfg.multiprocessing_distributed

        ngpus_per_node = torch.cuda.device_count()
        if self.eval_cfg.multiprocessing_distributed:
            self.eval_cfg.world_size = ngpus_per_node * self.eval_cfg.world_size
            mp.spawn(self.eval_main, nprocs=ngpus_per_node, args=(ngpus_per_node, self.eval_cfg))
        else:
            self.eval_main(self.eval_cfg.gpu, ngpus_per_node, self.eval_cfg)


def load_function_from_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class SpatialMoCov3(torch.nn.Module):
    def __init__(self, base_encoder_wrapper, base_encoder, dim=256, mlp_dim=4096, T=1.0, args=None, positions=None, layernames=None):
        super().__init__()
        self.model = base_encoder_wrapper(base_encoder, dim, mlp_dim, T)
        self.positions = positions
        self.args = args
        self.iteration_counter = 0

        self.intermediate_outputs = {}

        self.scaler = torch.cuda.amp.GradScaler()

        # Register hooks on the intermediate layers of the model
        for layername in layernames:
            layer = resolve_sequential_module_from_str(self.model.base_encoder, layername)
            layer.register_forward_hook(self.get_hook_fn(layername))
            print(f"Registered hook for {layername}")

        run_name = args.name
        if args.wandb and is_master_process():
            wandb.init(
                project=args.name.split("_")[1],
                name=run_name,
            )
        
        save_dir = Path(args.output_dir) / "eval"
        save_dir.mkdir(parents=True, exist_ok=True)

    def get_hook_fn(self, layername):
        """Creates a hook function to capture the outputs of a specific layer."""
        def hook_fn(module, input, output):
            spatial_features = output[:, 1:].float()
            N, L, H = spatial_features.shape
            sL = int(math.sqrt(L))
            spatial_features = spatial_features.reshape(N, sL, sL, H).permute(0, 3, 1, 2)
            self.intermediate_outputs[layername] = spatial_features
        return hook_fn

    def forward(self, x1, x2, m):
        self.intermediate_outputs.clear()

        loss_accumulator = self.model.forward(x1, x2, m)

        spatial_outputs = {}
        for layername, spatial_features in self.intermediate_outputs.items(): 
            spatial_outputs[layername] = (spatial_features, self.positions[layername])

        return loss_accumulator, spatial_outputs

    def get_joint_loss(self, loss):
        if isinstance(loss, tuple):
            loss_accumulator, spatial_outputs = loss

            task_loss = loss_accumulator.clone()
            joint_loss = loss_accumulator
            
            spatial_losses = {}
            for layername, layer_output in spatial_outputs.items():
                features, pos = layer_output

                spatial_losses[layername] = spatial_correlation_loss(
                    features.cuda(),
                    pos.coordinates.cuda(),
                    pos.neighborhood_indices.cuda(),
                )

                if dist.get_world_size() > 1:
                    dist.all_reduce(spatial_losses[layername])
                spatial_losses[layername] = spatial_losses[layername] / dist.get_world_size()
                
                joint_loss += self.args.spatial_loss.spatial_weight[layername] * spatial_losses[layername]

            if self.iteration_counter % 10 == 0:
                if is_master_process():
                    serializable_spatial_losses = convert_to_serializable(spatial_losses)
                    print(json.dumps(serializable_spatial_losses))
                    save_spatial_losses_path = Path(self.args.output_dir) / "spatial_losses.json"
                    with open(save_spatial_losses_path, 'a') as f:
                        f.write(json.dumps(serializable_spatial_losses) + "\n")

                    if self.args.wandb:
                        wandb.define_metric("iter")
                        log_data = {
                            "task_loss": task_loss,
                            "joint_loss": joint_loss,
                            "spatial_loss": {
                                **spatial_losses
                            },
                            "iter": self.iteration_counter
                        }
                        wandb.log(log_data)
            
            self.iteration_counter += 1
        else:
            joint_loss = loss

        return joint_loss


"""
Code modified from the MoCo v3 public repository
"""
def main_worker(gpu, ngpus_per_node, args, positions, layernames):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = SpatialMoCov3(
        builder.MoCo_ViT,
        partial(vit_base, stop_grad_conv1=args.stop_grad_conv1),
        args.moco_dim, args.moco_mlp_dim, args.moco_t, args.spatial_args, positions, layernames)

    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm

    if args.optimizer == 'lars':
        optimizer = optimizer.LARS(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
        
    scaler = torch.cuda.amp.GradScaler()
    summary_writer = SummaryWriter() if args.rank == 0 else None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([loader.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.ImageFolder(
        traindir,
        loader.TwoCropsTransform(transforms.Compose(augmentation1), 
                                    transforms.Compose(augmentation2)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, scaler, summary_writer, epoch, args)

        if (epoch % 25 == 0 or epoch == args.epochs - 1) and (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0)): # only the first GPU saves checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best=False, filename=args.save_dir / ('checkpoint_%04d.pth.tar' % epoch))

    if args.rank == 0:
        summary_writer.close()
