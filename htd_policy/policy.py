import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
from detr.main import build_ACT_model_and_optimizer, build_MXT_model_and_optimizer

from collections import OrderedDict

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'HIT':
        policy = HITPolicy(policy_config)
    elif policy_class == 'MXT':
        policy = MXTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'HIT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'MXT':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def _normalize_camera_images(image):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if isinstance(image, dict):
        return {name: normalize(frames) for name, frames in image.items()}
    return normalize(image)


class HITPolicy(nn.Module):
    """Non-autoregressive decoder-only ACT policy."""

    def __init__(self, args_override):
        super().__init__()
        args_override['model_type'] = "HIT"
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.feature_loss_weight = args_override['feature_loss_weight'] if 'feature_loss_weight' in args_override else 0.0
        try:
            self.state_idx = args_override['state_idx']
            self.action_idx = args_override['action_idx']
        except:
            self.state_idx = None
            self.action_idx = None
        self.precomputed_features = args_override['precomputed_features']
    
    def __call__(self, qpos, image, actions=None, is_pad=None, action_masks=None):
        if self.state_idx is not None:
            qpos = qpos[:, self.state_idx]
        if self.action_idx is not None and actions is not None:
            actions = actions[:, :, self.action_idx]
        
        if not self.precomputed_features:
            image = _normalize_camera_images(image)
        if actions is not None: # training/validation time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            loss_dict = dict()
            a_hat, _, hs_img_dict = self.model(qpos, image)
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            all_l1 = all_l1 * action_masks.unsqueeze(1)
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).sum() / (~is_pad).sum()
            loss_dict['l1'] = l1
            if self.model.feature_loss and self.model.training:
                loss_dict['feature_loss'] = F.mse_loss(hs_img_dict['hs_img'], hs_img_dict['src_future']).mean()
                loss_dict['loss'] = loss_dict['l1'] + self.feature_loss_weight*loss_dict['feature_loss']
            else:
                loss_dict['loss'] = loss_dict['l1']
            return loss_dict
        else: # inference time
            a_hat, _, _ = self.model(qpos, image) # no action, sample from prior
            return a_hat

    def forward_inf(self, qpos, image):
        if self.state_idx is not None:
            qpos = qpos[:, self.state_idx]
        
        image = _normalize_camera_images(image)
        a_hat, _, _ = self.model(qpos, image) # no action, sample from prior
        return a_hat
        
    def configure_optimizers(self):
        return self.optimizer   
        
    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        args_override['model_type'] = "ACT"
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.vq = args_override['vq']
        print(f'KL Weight {self.kl_weight}')
        try:
            self.state_idx = args_override['state_idx']
            self.action_idx = args_override['action_idx']
        except:
            self.state_idx = None
            self.action_idx = None
        self.precomputed_features = args_override['precomputed_features']
    

    def __call__(self, qpos, image, actions=None, is_pad=None, action_masks=None, vq_sample=None):
        if self.state_idx is not None:
            qpos = qpos[:, self.state_idx]
        if self.action_idx is not None and actions is not None:
            actions = actions[:, :, self.action_idx]
            
        env_state = None
        if not self.precomputed_features:
            image = _normalize_camera_images(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            loss_dict = dict()
            a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.model(qpos, image, env_state, actions, is_pad, vq_sample)
            if self.vq or self.model.encoder is None:
                total_kld = [torch.tensor(0.0)]
            else:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            if self.vq:
                loss_dict['vq_discrepancy'] = F.l1_loss(probs, binaries, reduction='mean')
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            all_l1 = all_l1 * action_masks.unsqueeze(1)
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).sum() / (~is_pad).sum()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _), _, _ = self.model(qpos, image, env_state, vq_sample=vq_sample) # no action, sample from prior
            return a_hat

    def forward_inf(self, qpos, image):
        if self.state_idx is not None:
            qpos = qpos[:, self.state_idx]
        env_state = None
        image = _normalize_camera_images(image)
        a_hat, _, (_, _), _, _ = self.model(qpos, image, env_state, vq_sample=None) # no action, sample from prior
        return a_hat
        
    def configure_optimizers(self):
        return self.optimizer

    @torch.no_grad()
    def vq_encode(self, qpos, actions, is_pad):
        actions = actions[:, :self.model.num_queries]
        is_pad = is_pad[:, :self.model.num_queries]

        _, _, binaries, _, _ = self.model.encode(qpos, actions, is_pad)

        return binaries
        
    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
    
    
class MXTPolicy(nn.Module):
    _CHECKPOINT_COMPONENTS = OrderedDict((
        ('tokenizers', 'tokenizers'),
        ('action_heads', 'action_heads'),
        ('dream_heads', 'dream_heads'),
        ('ema_teachers', 'ema_teachers'),
        ('dream_normalizers', 'dream_normalizers'),
    ))

    def __init__(self, args_override):
        super().__init__()
        args_override['model_type'] = "mxt"
        model, optimizer = build_MXT_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.share_cross_attn = args_override['share_cross_attn']
        
    def forward(self, data, embodiment, obs_mask=None, act_mask=None, infer_dream=False):
        return self.model(data, embodiment, obs_mask, act_mask, infer_dream=infer_dream)
    
    def compute_loss(self, batch):
        return self.model.compute_loss(batch)

    def post_step(self):
        """Called after optimizer.step() to update EMA teacher parameters."""
        self.model.update_ema()

    def configure_optimizers(self):
        return self.optimizer
    
    def serialize(self):
        tokenizers = self.model.tokenizers
        tokenizer_keys = list(tokenizers.keys())
        tokenizer_state_dicts = {key: tokenizers[key].state_dict() for key in tokenizer_keys}
        trunk_state_dict = self.model.transformer_body.state_dict()
        action_heads = self.model.action_heads
        action_head_keys = list(action_heads.keys())
        action_head_state_dicts = {key: action_heads[key].state_dict() for key in action_head_keys}
        result = {
            'tokenizers': tokenizer_state_dicts,
            'trunk': trunk_state_dict,
            'action_heads': action_head_state_dicts,
        }
        if self.model.dream_heads:
            dream_heads = self.model.dream_heads
            result['dream_heads'] = {key: dream_heads[key].state_dict() for key in dream_heads.keys()}
        if self.model.ema_teachers:
            ema_teachers = self.model.ema_teachers
            result['ema_teachers'] = {key: ema_teachers[key].state_dict() for key in ema_teachers.keys()}
        if self.model.dream_normalizers:
            dream_normalizers = self.model.dream_normalizers
            result['dream_normalizers'] = {key: dream_normalizers[key].state_dict() for key in dream_normalizers.keys()}
        return result
    
    def _checkpoint_module_dicts(self):
        result = OrderedDict()
        for checkpoint_key, attribute_name in self._CHECKPOINT_COMPONENTS.items():
            modules = getattr(self.model, attribute_name, None)
            if modules is not None:
                result[checkpoint_key] = modules
        return result

    @staticmethod
    def _state_dict_errors(module, checkpoint_state, label, *, require_dtype=True):
        if not isinstance(checkpoint_state, dict):
            return [f"{label} must be a state-dict mapping, got {type(checkpoint_state).__name__}"]
        expected_state = module.state_dict()
        expected_keys = set(expected_state.keys())
        checkpoint_keys = set(checkpoint_state.keys())
        errors = []
        missing = sorted(expected_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - expected_keys)
        if missing:
            errors.append(f"{label} missing parameter keys: {missing}")
        if unexpected:
            errors.append(f"{label} has unexpected parameter keys: {unexpected}")
        value_mismatches = []
        for key in sorted(expected_keys & checkpoint_keys):
            expected_value = expected_state[key]
            checkpoint_value = checkpoint_state[key]
            if isinstance(expected_value, torch.Tensor) and not isinstance(checkpoint_value, torch.Tensor):
                value_mismatches.append(
                    f"{key}: expected Tensor, "
                    f"got {type(checkpoint_value).__name__}"
                )
            elif isinstance(expected_value, torch.Tensor):
                if tuple(expected_value.shape) != tuple(checkpoint_value.shape):
                    value_mismatches.append(
                        f"{key}: expected shape {tuple(expected_value.shape)}, "
                        f"got {tuple(checkpoint_value.shape)}"
                    )
                if require_dtype and expected_value.dtype != checkpoint_value.dtype:
                    value_mismatches.append(
                        f"{key}: expected dtype {expected_value.dtype}, "
                        f"got {checkpoint_value.dtype}"
                    )
                if checkpoint_value.is_meta:
                    value_mismatches.append(f"{key}: checkpoint tensor is on the meta device")
                if checkpoint_value.layout != torch.strided:
                    value_mismatches.append(
                        f"{key}: expected strided tensor layout, "
                        f"got {checkpoint_value.layout}"
                    )
            elif type(expected_value) is not type(checkpoint_value):
                value_mismatches.append(
                    f"{key}: expected {type(expected_value).__name__}, "
                    f"got {type(checkpoint_value).__name__}"
                )
        if value_mismatches:
            errors.append(f"{label} has value mismatches: {value_mismatches}")
        return errors

    def _strict_checkpoint_errors(self, model_dicts, module_dicts):
        if not isinstance(model_dicts, dict):
            return [f"checkpoint must be a mapping, got {type(model_dicts).__name__}"]

        expected_top_level = {'tokenizers', 'trunk', 'action_heads'}
        expected_top_level.update(
            key for key, modules in module_dicts.items()
            if key not in {'tokenizers', 'action_heads'} and len(modules) > 0
        )
        checkpoint_top_level = set(model_dicts.keys())
        errors = []
        missing_top_level = sorted(expected_top_level - checkpoint_top_level)
        unexpected_top_level = sorted(checkpoint_top_level - expected_top_level)
        if missing_top_level:
            errors.append(f"missing top-level component keys: {missing_top_level}")
        if unexpected_top_level:
            errors.append(f"unexpected top-level component keys: {unexpected_top_level}")

        if 'trunk' in model_dicts:
            errors.extend(self._state_dict_errors(
                self.model.transformer_body, model_dicts['trunk'], 'trunk'))

        for component, modules in module_dicts.items():
            if component not in expected_top_level or component not in model_dicts:
                continue
            checkpoint_modules = model_dicts[component]
            if not isinstance(checkpoint_modules, dict):
                errors.append(
                    f"{component} must map embodiment keys to state dicts, "
                    f"got {type(checkpoint_modules).__name__}"
                )
                continue
            expected_keys = set(modules.keys())
            checkpoint_keys = set(checkpoint_modules.keys())
            missing = sorted(expected_keys - checkpoint_keys)
            unexpected = sorted(checkpoint_keys - expected_keys)
            if missing:
                errors.append(f"{component} missing embodiment keys: {missing}")
            if unexpected:
                errors.append(f"{component} has unexpected embodiment keys: {unexpected}")
            for key in sorted(expected_keys & checkpoint_keys):
                errors.extend(self._state_dict_errors(
                    modules[key], checkpoint_modules[key], f"{component}[{key!r}]"))
        return errors

    @staticmethod
    def _load_component(module, checkpoint_state, label):
        try:
            module.load_state_dict(checkpoint_state, strict=True)
        except (RuntimeError, TypeError) as exc:
            raise RuntimeError(f"Failed to load MXT checkpoint component {label}: {exc}") from exc

    def deserialize(
        self,
        model_dicts,
        eval=False,
        *,
        mode=None,
        transfer_source_embodiment=None,
    ):
        """Load an MXT checkpoint in strict-artifact or partial-transfer mode.

        ``mode='strict'`` requires exactly the components and embodiment keys
        produced by :meth:`serialize` for this model.  It preflights parameter
        names and shapes before changing any weights and is intended for
        rollout and exact training resume.

        ``mode='transfer'`` loads the trunk plus every matching tokenizer,
        action head, dream head, EMA teacher, and dream normalizer.  Missing or
        extra embodiment keys are reported rather than treated as an exact
        artifact match.  If ``share_cross_attn`` is enabled, cross-attention
        weights may additionally be copied from one checkpoint embodiment into
        tokenizer embodiments absent from the checkpoint.

        The legacy ``eval=True`` spelling selects strict mode.  Legacy
        ``eval=False`` (the default) selects transfer mode, but unlike the old
        implementation it does load all matching tokenizer and action-head
        weights instead of silently leaving them randomly initialized.
        """
        if mode is None:
            mode = 'strict' if eval else 'transfer'
        elif eval and mode != 'strict':
            raise ValueError("eval=True is an alias for mode='strict'; conflicting mode supplied")
        if mode not in {'strict', 'transfer'}:
            raise ValueError("MXT deserialize mode must be 'strict' or 'transfer'")
        if not isinstance(model_dicts, dict):
            raise TypeError(f"MXT checkpoint must be a mapping, got {type(model_dicts).__name__}")

        module_dicts = self._checkpoint_module_dicts()
        if mode == 'strict':
            errors = self._strict_checkpoint_errors(model_dicts, module_dicts)
            if errors:
                detail = '\n  - '.join(errors)
                raise RuntimeError(f"Strict MXT artifact load failed:\n  - {detail}")

            report = {'mode': 'strict', 'loaded': {'trunk': ['<root>']}}
            self._load_component(self.model.transformer_body, model_dicts['trunk'], 'trunk')
            for component, modules in module_dicts.items():
                if component not in model_dicts:
                    continue
                loaded_keys = []
                for key in modules.keys():
                    self._load_component(
                        modules[key], model_dicts[component][key],
                        f"{component}[{key!r}]",
                    )
                    loaded_keys.append(key)
                report['loaded'][component] = loaded_keys
            return report

        required = {'tokenizers', 'trunk', 'action_heads'}
        missing_required = sorted(required - set(model_dicts.keys()))
        if missing_required:
            raise RuntimeError(
                "MXT transfer checkpoint is missing required top-level component keys: "
                f"{missing_required}"
            )
        if not isinstance(model_dicts['tokenizers'], dict):
            raise TypeError("MXT checkpoint tokenizers must map embodiment keys to state dicts")
        if not isinstance(model_dicts['action_heads'], dict):
            raise TypeError("MXT checkpoint action_heads must map embodiment keys to state dicts")

        known_top_level = {'trunk'} | set(module_dicts.keys())
        report = {
            'mode': 'transfer',
            'loaded': {'trunk': ['<root>']},
            'missing_model_keys': {},
            'ignored_checkpoint_keys': {},
            'cross_attention_copied': {},
            'ignored_top_level_keys': sorted(set(model_dicts.keys()) - known_top_level),
        }

        transfer_errors = self._state_dict_errors(
            self.model.transformer_body,
            model_dicts['trunk'],
            'trunk',
            require_dtype=False,
        )
        component_plans = []
        for component, modules in module_dicts.items():
            checkpoint_modules = model_dicts.get(component, {})
            if not isinstance(checkpoint_modules, dict):
                raise TypeError(
                    f"MXT checkpoint {component} must map embodiment keys to state dicts"
                )
            model_keys = set(modules.keys())
            checkpoint_keys = set(checkpoint_modules.keys())
            matching_keys = sorted(model_keys & checkpoint_keys)
            for key in matching_keys:
                transfer_errors.extend(self._state_dict_errors(
                    modules[key],
                    checkpoint_modules[key],
                    f"{component}[{key!r}]",
                    require_dtype=False,
                ))
            report['loaded'][component] = matching_keys
            missing = sorted(model_keys - checkpoint_keys)
            ignored = sorted(checkpoint_keys - model_keys)
            if missing:
                report['missing_model_keys'][component] = missing
            if ignored:
                report['ignored_checkpoint_keys'][component] = ignored
            component_plans.append((component, modules, checkpoint_modules, matching_keys))

        missing_tokenizers = report['missing_model_keys'].get('tokenizers', [])
        cross_attention_plan = None
        if missing_tokenizers and self.share_cross_attn:
            checkpoint_tokenizers = model_dicts['tokenizers']
            available_sources = sorted(checkpoint_tokenizers.keys())
            if transfer_source_embodiment is None:
                if len(available_sources) != 1:
                    raise ValueError(
                        "Partial MXT cross-embodiment transfer has multiple source "
                        f"tokenizers {available_sources}; set transfer_source_embodiment explicitly"
                    )
                transfer_source_embodiment = available_sources[0]
            if transfer_source_embodiment not in checkpoint_tokenizers:
                raise ValueError(
                    f"transfer_source_embodiment {transfer_source_embodiment!r} is not in "
                    f"checkpoint tokenizers {available_sources}"
                )

            source_state = checkpoint_tokenizers[transfer_source_embodiment]
            if not isinstance(source_state, dict):
                raise TypeError(
                    "MXT transfer source tokenizer must be a state-dict mapping, "
                    f"got {type(source_state).__name__}"
                )
            cross_attention_names = sorted(
                name for name in source_state.keys() if 'cross_attn' in name
            )
            with torch.no_grad():
                for target_key in missing_tokenizers:
                    target_state = self.model.tokenizers[target_key].state_dict()
                    incompatible = []
                    for name in cross_attention_names:
                        if name not in target_state:
                            incompatible.append(f"missing {name!r}")
                            continue
                        source_value = source_state[name]
                        target_value = target_state[name]
                        if not isinstance(source_value, torch.Tensor):
                            incompatible.append(
                                f"{name!r} is {type(source_value).__name__}, not Tensor"
                            )
                        elif tuple(target_value.shape) != tuple(source_value.shape):
                            incompatible.append(
                                f"{name!r} shape {tuple(source_value.shape)} -> "
                                f"{tuple(target_value.shape)}"
                            )
                        else:
                            try:
                                torch.empty_like(target_value).copy_(source_value)
                            except (RuntimeError, TypeError) as exc:
                                incompatible.append(f"{name!r} cannot be copied: {exc}")
                    if incompatible:
                        transfer_errors.append(
                            "Cannot copy MXT cross-attention weights from "
                            f"{transfer_source_embodiment!r} to {target_key!r}: "
                            f"{incompatible}"
                        )

            cross_attention_plan = (
                transfer_source_embodiment,
                source_state,
                cross_attention_names,
                missing_tokenizers,
            )

        if transfer_errors:
            detail = '\n  - '.join(transfer_errors)
            raise RuntimeError(f"MXT transfer checkpoint preflight failed:\n  - {detail}")

        self._load_component(self.model.transformer_body, model_dicts['trunk'], 'trunk')
        for component, modules, checkpoint_modules, matching_keys in component_plans:
            for key in matching_keys:
                self._load_component(
                    modules[key], checkpoint_modules[key], f"{component}[{key!r}]"
                )

        if cross_attention_plan is not None:
            source_embodiment, source_state, cross_attention_names, target_keys = (
                cross_attention_plan
            )
            with torch.no_grad():
                for target_key in target_keys:
                    target_state = self.model.tokenizers[target_key].state_dict()
                    for name in cross_attention_names:
                        target_state[name].copy_(source_state[name])
                    report['cross_attention_copied'][target_key] = cross_attention_names
        return report
