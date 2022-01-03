import sys
import torch.nn as nn
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class HFTransformer(nn.Module):
    def __init__(self, pretrained_checkpoint, pretrained=False):
        from transformers import AutoConfig, AutoModelForImageClassification

        super().__init__()
        self.pretrained_checkpoint = pretrained_checkpoint
        if pretrained:
            self.net = AutoModelForImageClassification.from_pretrained(pretrained_checkpoint)
        else:
            _config = AutoConfig.from_pretrained(pretrained_checkpoint)
            self.net = AutoModelForImageClassification.from_config(_config)
        config = self.net.config
        self.default_cfg = {
            'num_classes': self.net.num_labels,
            'input_size': (config.num_channels, config.image_size, config.image_size),
            'mean': (0.5, 0.5, 0.5),
            'std': (0.5, 0.5, 0.5),
            'classifier': 'classifier'}
        if 'facebook/deit' in pretrained_checkpoint:
            self.default_cfg['mean'] = IMAGENET_DEFAULT_MEAN
            self.default_cfg['std'] = IMAGENET_DEFAULT_STD
            # self.default_cfg['classifier'] = ('cls_classifier', 'distillation_classifier')

    def load_state_dict(self, state_dict):
        return self.net.load_state_dict(state_dict)

    def state_dict(self):
        return self.net.state_dict()

    def __setattr__(self, name, value):
        # override setter method for nn.Module
        # ignore call of nn.Module.__setattr__ when setting classifier
        if name == 'classifier':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    @property
    def classifier(self):
        if hasattr(self.net, 'classifier'):
            out = self.net.classifier
        elif hasattr(self.net, 'cls_classifier') and hasattr(self.net, 'distillation_classifier'):
            out = self.net.cls_classifier
        else:
            raise ValueError(
                f'Unknown classifier in HF Vision Transformer: {self.pretrained_checkpoint}')
        return out

    @classifier.setter
    def classifier(self, val):
        if hasattr(self.net, 'classifier'):
            self.net.classifier = val
        elif hasattr(self.net, 'cls_classifier') and hasattr(self.net, 'distillation_classifier'):
            self.net.cls_classifier = val
            self.net.distillation_classifier = val
        else:
            raise ValueError(
                f'Unknown classifier in HF Vision Transformer: {self.pretrained_checkpoint}')

    def forward(self, x):
        return self.net(x).logits


class _FunctionWrapper:
    def __init__(self, func, func_name):
        self.func = func
        self.func.__name__ = func_name
        self.__name__ = func_name

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __str__(self):
        return self.func.__name__

    def __repr__(self):
        return str(self)


def _model_factory(arch_name, pretrained_checkpoint, use_hf=False):
    def create_model(out_dim=None, pretrained=False, *args, **kwargs):
        if use_hf:
            model = HFTransformer(pretrained_checkpoint, pretrained=pretrained)
        else:
            model = timm.create_model(pretrained_checkpoint, pretrained=pretrained, *args, **kwargs)

        assert model.default_cfg['num_classes'] == 1000, 'The checkpoint is not ImageNet-1k pre-trained.'
        input_size = model.default_cfg.get('test_input_size', model.default_cfg['input_size'])
        model.pretrained_config = {
            'pretrained_checkpoint': pretrained_checkpoint,
            'input_size': input_size[-1],
            'image_mean': model.default_cfg['mean'],
            'image_std': model.default_cfg['std']}

        # set custom classification head
        if out_dim is not None:
            classifier_attr = model.default_cfg['classifier']
            # some models like distilled transformers have multiple heads
            if isinstance(classifier_attr, str):
                classifier_attr = (classifier_attr,)
            for attr in classifier_attr:
                in_features = getattr(model, attr).in_features
                setattr(model, attr, nn.Linear(in_features, out_dim))

        return model
    return _FunctionWrapper(create_model, arch_name)


_timm_model_specs = {
    'resnet50': 'tv_resnet50',
    'resnet101': 'tv_resnet101',

    'resnext50': 'resnext50_32x4d',
    'resnext101': 'ig_resnext101_32x8d',

    'resnest26': 'resnest26d',
    'resnest50': 'resnest50d',
    'resnest101': 'resnest101e',
    'resnest200': 'resnest200e',
    'resnest269': 'resnest269e',

    'seresnext101': 'legacy_seresnext101_32x4d',

    'efficientnet_b0': 'tf_efficientnet_b0',
    'efficientnet_b1': 'tf_efficientnet_b1',
    'efficientnet_b2': 'tf_efficientnet_b2',
    'efficientnet_b3': 'tf_efficientnet_b3',
    'efficientnet_b4': 'tf_efficientnet_b4',
    'efficientnet_b5': 'tf_efficientnet_b5',
    'efficientnet_b6': 'tf_efficientnet_b6',
    'efficientnet_b7': 'tf_efficientnet_b7',

    'efficientnet_b0_ns': 'tf_efficientnet_b0_ns',
    'efficientnet_b1_ns': 'tf_efficientnet_b1_ns',
    'efficientnet_b2_ns': 'tf_efficientnet_b2_ns',
    'efficientnet_b3_ns': 'tf_efficientnet_b3_ns',
    'efficientnet_b4_ns': 'tf_efficientnet_b4_ns',
    'efficientnet_b5_ns': 'tf_efficientnet_b5_ns',
    'efficientnet_b6_ns': 'tf_efficientnet_b6_ns',
    'efficientnet_b7_ns': 'tf_efficientnet_b7_ns',
    
    'efficientnetv2_b0': 'tf_efficientnetv2_b0',
    'efficientnetv2_b1': 'tf_efficientnetv2_b1',
    'efficientnetv2_b2': 'tf_efficientnetv2_b2',
    'efficientnetv2_b3': 'tf_efficientnetv2_b3',
    'efficientnetv2_s': 'tf_efficientnetv2_s',
    'efficientnetv2_m': 'tf_efficientnetv2_m',
    'efficientnetv2_l': 'tf_efficientnetv2_l',

    # 'vit_tiny_224': 'vit_tiny_patch16_224',
    # 'vit_tiny_384': 'vit_tiny_patch16_384',
    # 'vit_small_224': 'vit_small_patch16_224',
    # 'vit_small_384': 'vit_small_patch16_384',
    # 'vit_base_224': 'vit_base_patch16_224',
    # 'vit_base_384': 'vit_base_patch16_384',
    # 'vit_base_patch32_224': 'vit_base_patch32_224',
    # 'vit_base_patch32_384': 'vit_base_patch32_384',
    # 'vit_large_224': 'vit_large_patch16_224',
    # 'vit_large_384': 'vit_large_patch16_384',

    # 'deit_tiny_224': 'deit_tiny_patch16_224',
    # 'deit_small_224': 'deit_small_patch16_224',
    # 'deit_base_224': 'deit_base_patch16_224',
    # 'deit_base_384': 'deit_base_patch16_384',
    # 'deit_tiny_distilled_224': 'deit_tiny_distilled_patch16_224',
    # 'deit_small_distilled_224': 'deit_small_distilled_patch16_224',
    # 'deit_base_distilled_224': 'deit_base_distilled_patch16_224',
    # 'deit_base_distilled_384': 'deit_base_distilled_patch16_384',
    
    # 'beit_base_224': 'beit_base_patch16_224',
    # 'beit_base_384': 'beit_base_patch16_384',
    # 'beit_large_224': 'beit_large_patch16_224',
    # 'beit_large_384': 'beit_large_patch16_384',
    # 'beit_large_512': 'beit_large_patch16_512',
}


_hf_model_specs = {
    'vit_base_224': 'google/vit-base-patch16-224',
    'vit_base_384': 'google/vit-base-patch16-384',
    'vit_base_patch32_224': 'google/vit-base-patch32-224',
    'vit_base_patch32_384': 'google/vit-base-patch32-384',
    'vit_large_224': 'google/vit-large-patch16-224',
    'vit_large_384': 'google/vit-large-patch16-384',

    'deit_tiny_224': 'facebook/deit-tiny-patch16-224',
    'deit_small_224': 'facebook/deit-small-patch16-224',
    'deit_base_224': 'facebook/deit-base-patch16-224',
    'deit_base_384': 'facebook/deit-base-patch16-384',
    'deit_tiny_distilled_224': 'facebook/deit-tiny-distilled-patch16-224',
    'deit_small_distilled_224': 'facebook/deit-small-distilled-patch16-224',
    'deit_base_distilled_224': 'facebook/deit-base-distilled-patch16-224',
    'deit_base_distilled_384': 'facebook/deit-base-distilled-patch16-384',

    'beit_base_224': 'microsoft/beit-base-patch16-224',
    'beit_base_384': 'microsoft/beit-base-patch16-384',
    'beit_large_224': 'microsoft/beit-large-patch16-224',
    'beit_large_384': 'microsoft/beit-large-patch16-384',
    'beit_large_512': 'microsoft/beit-large-patch16-512',
}


# create dictionary with models
MODELS = {
    **{k: _model_factory(k, v) for k, v in _timm_model_specs.items()},
    **{k: _model_factory(k, v, use_hf=True) for k, v in _hf_model_specs.items()}}


# add models as attribute of this module
for k, v in MODELS.items():
    setattr(sys.modules[__name__], k, v)


def get_model_fn(model_arch):
    if callable(model_arch):
        model_fn = model_arch
    elif model_arch in MODELS:
        model_fn = MODELS[model_arch]
    else:
        raise ValueError(f'Unknown model architecture "{model_arch}".')
    return model_fn


def get_model(model_arch, out_dim=None, pretrained=True, *args, **kwargs):
    model_fn = get_model_fn(model_arch)
    model = model_fn(out_dim, pretrained, *args, **kwargs)
    return model
