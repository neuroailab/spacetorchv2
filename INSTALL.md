# Installation


## 0. Requirements

- Linux, CUDA 11.7-compatible GPU driver
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda

## 1. Create the conda environment

```bash
conda env create -f conda.yml
conda activate spacetorch
```


## 2. Install pure-Python pip requirements

```bash
pip install -r requirements.txt
```

## 3. Install packages from source


```bash
pip install classy-vision==0.7.0
pip install fairscale==0.4.13

cd ..

git clone https://github.com/facebookresearch/vissl
cd vissl
pip install --progress-bar off -r requirements.txt
pip install -e .
cd ..

git clone https://github.com/facebookresearch/dinov2
cd dinov2
pip install -e .
cd ..

git clone https://github.com/facebookresearch/dino
git clone https://github.com/facebookresearch/moco-v3
git clone https://github.com/SwinTransformer/Transformer-SSL
git clone https://github.com/facebookresearch/mae

cd spacetorchv2
pip install -e .
```

# For model training only, the following changes need to be made to the above downloaded repositories

## DINO-v2

1- Add to `dinov2/configs/train`:

`vitb14_short.yaml`
```yaml
train:
  dataset_path: ImageNet:split=TRAIN:root=/path/to/dataset/:extra=/path/to/dataset/extra
  batch_size_per_gpu: 128
student:
  block_chunks: 0
  patch_size: 14
  arch: vit_base
crops:
  local_crops_size: 98
```

2- Modify `dinov2/eval/setup.py`:

```python
def build_model_for_eval(config, pretrained_weights):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    if pretrained_weights:
        dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    model.cuda()
    return model


def setup_and_build_model(args, distributed=True) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    config = setup(args, enable_distributed=distributed)
    model = build_model_for_eval(config, args.pretrained_weights)
    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype
```

3- Add to `dinov2/train/ssl_meta_arch.py`:
```python
def forward(self, images, teacher_temp):
    n_global_crops = 2
    assert n_global_crops == 2
    n_local_crops = self.cfg.crops.local_crops_number

    global_crops = images["collated_global_crops"].cuda(non_blocking=True)
    local_crops = images["collated_local_crops"].cuda(non_blocking=True)

    masks = images["collated_masks"].cuda(non_blocking=True)
    mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
    n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
    n_masked_patches = mask_indices_list.shape[0]
    upperbound = images["upperbound"]
    masks_weight = images["masks_weight"].cuda(non_blocking=True)

    n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
    n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

    do_dino = self.do_dino
    do_ibot = self.do_ibot

    # loss scales
    ibot_loss_scale = 1.0 / n_global_crops

    # teacher output
    @torch.no_grad()
    def get_teacher_output():
        x, n_global_crops_teacher = global_crops, n_global_crops
        teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
        teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
        teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
        # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
        teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
        ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
        _dim = ibot_teacher_patch_tokens.shape[-1]
        n_cls_tokens = teacher_cls_tokens.shape[0]

        if do_ibot and not self.ibot_separate_head:
            buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
            buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
            torch.index_select(
                ibot_teacher_patch_tokens.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
                out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
            )
            tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
            teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
            masked_teacher_patch_tokens_after_head = tokens_after_head[
                n_cls_tokens : n_cls_tokens + n_masked_patches
            ]
        elif do_ibot and self.ibot_separate_head:
            buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
            torch.index_select(
                ibot_teacher_patch_tokens.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
                out=buffer_tensor_teacher[:n_masked_patches],
            )
            teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
            masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                :n_masked_patches
            ]
        else:
            teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
            masked_teacher_ibot_softmaxed_centered = None

        if self.cfg.train.centering == "centering":
            teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                teacher_cls_tokens_after_head, teacher_temp=teacher_temp
            ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
            self.dino_loss.update_center(teacher_cls_tokens_after_head)
            if do_ibot:
                masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                    masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                )
                masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

        elif self.cfg.train.centering == "sinkhorn_knopp":
            teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                teacher_cls_tokens_after_head, teacher_temp=teacher_temp
            ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

            if do_ibot:
                masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                    masked_teacher_patch_tokens_after_head,
                    teacher_temp=teacher_temp,
                    n_masked_patches_tensor=n_masked_patches_tensor,
                )

        else:
            raise NotImplementedError

        return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

    teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
    reshard_fsdp_model(self.teacher)

    loss_dict = {}

    loss_accumulator = 0  # for backprop
    student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
        [global_crops, local_crops], masks=[masks, None], is_training=True
    )

    inputs_for_student_head_list = []

    # 1a: local crops cls tokens
    student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
    inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

    # 1b: global crops cls tokens
    student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
    inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

    # 1c: global crops patch tokens
    if do_ibot:
        _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
        ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
        buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
        buffer_tensor_patch_tokens[:n_masked_patches].copy_(
            torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
        )
        if not self.ibot_separate_head:
            inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
        else:
            student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                :n_masked_patches
            ]

    # 2: run
    _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
    outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

    # 3a: local crops cls tokens
    student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

    # 3b: global crops cls tokens
    student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

    # 3c: global crops patch tokens
    if do_ibot and not self.ibot_separate_head:
        student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

    if n_local_crops > 0:
        dino_local_crops_loss = self.dino_loss(
            student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
            teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
        ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

        # store for display
        loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

        # accumulate loss
        loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

    # process global crops
    loss_scales = 2  # this is here since we process global crops together

    if do_dino:
        # compute loss
        dino_global_crops_loss = (
            self.dino_loss(
                student_output_list=[student_global_cls_tokens_after_head],
                teacher_out_softmaxed_centered_list=[
                    teacher_dino_softmaxed_centered_list.flatten(0, 1)
                ],  # these were chunked and stacked in reverse so A is matched to B
            )
            * loss_scales
            / (n_global_crops_loss_terms + n_local_crops_loss_terms)
        )

        loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

        # accumulate loss
        loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

        student_cls_tokens = student_global_cls_tokens

        if self.do_koleo:
            koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
            )  # we don't apply koleo loss between cls tokens of a same image
            loss_accumulator += koleo_loss
            loss_dict["koleo_loss"] = (
                koleo_loss / loss_scales
            )  # this is to display the same losses as before but we can remove eventually

    if do_ibot:
        # compute loss
        ibot_patch_loss = (
            self.ibot_patch_loss.forward_masked(
                student_global_masked_patch_tokens_after_head,
                masked_teacher_ibot_softmaxed_centered,
                student_masks_flat=masks,
                n_masked_patches=n_masked_patches,
                masks_weight=masks_weight,
            )
            * loss_scales
            * ibot_loss_scale
        )

        # store for display
        loss_dict["ibot_loss"] = ibot_patch_loss / 2

        # accumulate loss
        loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

    return loss_accumulator, loss_dict
```

4- Modify `dinov2/train/train.py`:

Add `parser.add_argument("--local-rank", default=0, type=int, help="Variable for distributed computing.")`

Instead of calling `model.forward_backward(...)`, call
```python
loss_accumulator, loss_dict = model.forward(data, teacher_temp=teacher_temp)
model.backprop_loss(loss_accumulator)
model.fsdp_synchronize_streams()
```

5- Modify `dinov2/utils/config.py`:

```python
def default_setup(args, enable_distributed=True):
    if enable_distributed:
        distributed.enable(overwrite=True)
        rank = distributed.get_global_rank()
    else:
        rank = 0
        
    seed = getattr(args, "seed", 0) 

    global logger
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    utils.fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))


def setup(args, enable_distributed=True):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    default_setup(args, enable_distributed=enable_distributed)
    apply_scaling_rules_to_cfg(cfg)
    write_config(cfg, args.output_dir)
    return cfg
```

6- Track `self.attn_bias = attn_bias` in `MemEffAttention` class in `dinov2/layers/attention.py`.

## Transformer-SSL

1- Add to `configs/`:

`moby_swin_tiny_w{size}.yaml`
```yaml
TRAIN:
  WARMUP_EPOCHS: 5
  EPOCHS: 300
  BASE_LR: 0.001
  WEIGHT_DECAY: 0.05
AUG:
  SSL_AUG: True
MODEL:
  TYPE: moby
  NAME: moby__swin_tiny__patch4_window44_224__odpr02_tdpr0_cm099_ct02_queue4096_proj2_pred2
  SWIN:
    EMBED_DIM: 96
    DEPTHS: [ 2, 2, 6, 2 ]
    NUM_HEADS: [ 3, 6, 12, 24 ]
    WINDOW_SIZE: {window[size]}
  MOBY:
    ENCODER: swin
    ONLINE_DROP_PATH_RATE: 0.2
    TARGET_DROP_PATH_RATE: 0.0
    CONTRAST_MOMENTUM: 0.99
    CONTRAST_TEMPERATURE: 0.2
    CONTRAST_NUM_NEGATIVE: 4096
    PROJ_NUM_LAYERS: 2
    PRED_NUM_LAYERS: 2
```
where `size = {2,4,14,28,56}`
and `window[2] = [2, 2, 2, 7]`, `window[4] = [4, 4, 7, 7]`, `window[14] = [14, 14, 14, 7]`, `window[28] = [28, 28, 14, 7]`, `window[56] = [56, 28, 14, 7]`. The standard window size is `[7, 7, 7, 7]`

2- In `config.py`, modify `_C.AMP_OPT_LEVEL = 'O0'`

3- In `data/build.py`, modify `from timm.data.transforms import str_to_pil_interp`

4- In `models/build.py`, modify `from timm.models import deit_small_patch16_224` and `deit_small=deit_small_patch16_224`