import torch
from pathlib import Path
from collections import namedtuple

from spacetorch.variants.base_arch import BaseArch
from spacetorch.variants.positions import NetworkPositions
from spacetorch.variants.assets.lcnn.resnet_imagenet_continuoustopo import ResNet18


class LCNN(BaseArch):
    """
    Implements the LLCNN architecture (Qian et al., 2024).
    """
    def __init__(self):
        super().__init__()

    def get_model_layernames(self, args):
        return list(args.spatial_loss.spatial_weight.keys())

    def set_cfg(self, args):
        pass

    def set_eval_cfg(self, args):
        pass

    def _load_pretrained_weights(self, args, model):
        state_dict = torch.load(args.variant.params.pretrained_weights, map_location="cpu")
        state_dict['state_dict'] = {k.replace('module.', ''): state_dict['state_dict'][k] for k in state_dict['state_dict'].keys()}
        msg = model.load_state_dict(state_dict['state_dict'], strict=False)
        print(("Pretrained weights found at {} and loaded with msg: {}".format(args.variant.params.pretrained_weights, msg)))

    def _load_positions(self, position_dir):
        network_positions = NetworkPositions.load_from_dir(position_dir)
        network_positions.to_torch()
        return network_positions.layer_positions

    def set_model(self, args):
        pass

    def set_eval_model(self, args):
        self.eval_model = load_model(pool_type='gaussian', kap_kernelsize=0.23, continuous=True, local_conv=False)
        if args.variant.params.pretrained_weights:
            self._load_pretrained_weights(args, self.eval_model)
        self.eval_model.cuda()
        self.eval_model.eval()

    def start_training_protocol(self, args):
        pass

    def set_eval_protocol(self, args):
        pass

    def start_eval_protocol(self):
        pass

    def set_kinetics_cfg(self, args):
        pass

    def set_kinetics_protocol(self, args):
        pass
    
    def start_kinetics_protocol(self):
        pass


def load_model(pool_type, kap_kernelsize, continuous, local_conv):
    Args = namedtuple('nt', ['dataset', 'arch', 'pool_type', 'max_num_pools', 'noise_std', 
                             'kap_kernelsize', 'kap_stride', 'expansion', 'do_prob', 
                             'continuous', 'local_conv'])
    
    args = Args(dataset="imagenet", arch="resnet18contopo", pool_type=pool_type, 
                max_num_pools=1, noise_std=0., kap_kernelsize=kap_kernelsize, kap_stride=1, 
                expansion=1, do_prob=0., continuous=continuous, local_conv=local_conv)
    
    model = ResNet18(1000, args.pool_type, args.max_num_pools, args.noise_std, args.kap_kernelsize, args.continuous, args.local_conv)

    return model
