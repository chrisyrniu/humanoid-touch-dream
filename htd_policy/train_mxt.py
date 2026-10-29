import fcntl
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import numpy as np
import os
import pickle
import argparse
from itertools import repeat
from tqdm import tqdm
import json
import wandb
from mxt_utils import (
    HTD_POLICY_EMBODIMENT,
    KEY_PATH_MAP_G1_INSPIRE_WB,
    load_data,
    find_all_hdf5,
    compute_dict_mean,
    set_seed,
) # data functions
from policy import make_policy, make_optimizer
import yaml
import gc
import contextlib
from copy import deepcopy


AUTO_RESUME_SNAPSHOT = 'training_last.ckpt'


def _effective_ckpt_dir(args, experiment_suffix=None):
    suffix = args['policy_class'] if experiment_suffix is None else experiment_suffix
    return f'{args["ckpt_dir"]}_{args["task_name"]}_{suffix}'


def _configure_auto_resume(args, experiment_suffix=None):
    """Point a restarted command at its own stable output directory."""
    if not args.get('auto_resume'):
        return
    target_dir = _effective_ckpt_dir(args, experiment_suffix)
    if not os.path.isdir(target_dir):
        return
    snapshot_path = os.path.join(target_dir, AUTO_RESUME_SNAPSHOT)
    policy_checkpoint_path = os.path.join(target_dir, 'policy_last.ckpt')
    if os.path.isfile(snapshot_path) or os.path.isfile(policy_checkpoint_path):
        previous_source = args.get('resume_training')
        if previous_source and os.path.realpath(previous_source) != os.path.realpath(target_dir):
            print(f'Auto-resume supersedes initial resume source {previous_source}')
        args['resume_training'] = target_dir
        print(f'Auto-resume found existing run: {target_dir}')
    else:
        print(
            f'Auto-resume found no committed checkpoint in {target_dir}; '
            'restarting initialization in the existing directory'
        )


def _atomic_torch_save(value, destination):
    temporary = f'{destination}.tmp-{os.getpid()}'
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json_dump(value, destination, **dump_kwargs):
    temporary = f'{destination}.tmp-{os.getpid()}'
    try:
        with open(temporary, 'w') as stream:
            json.dump(value, stream, **dump_kwargs)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _training_state(step, min_val_loss):
    state = {
        'step': step,
        'min_val_loss': float(min_val_loss) if np.isfinite(min_val_loss) else 1e30,
    }
    if wandb.run is not None:
        state['wandb_run_id'] = wandb.run.id
    return state


def _save_resume_checkpoint(policy, optimizer, ckpt_dir, step, min_val_loss):
    """Commit one coherent resume bundle before refreshing legacy artifacts."""
    model_state = policy.serialize()
    optimizer_state = optimizer.state_dict()
    state = _training_state(step, min_val_loss)
    snapshot = {
        'format_version': 1,
        'policy': model_state,
        'optimizer': optimizer_state,
        'training_state': state,
    }
    _atomic_torch_save(snapshot, os.path.join(ckpt_dir, AUTO_RESUME_SNAPSHOT))
    _atomic_torch_save(model_state, os.path.join(ckpt_dir, 'policy_last.ckpt'))
    _atomic_torch_save(optimizer_state, os.path.join(ckpt_dir, 'optimizer_last.ckpt'))
    _atomic_json_dump(state, os.path.join(ckpt_dir, 'training_state.json'))

def forward_pass(batch, policy):
    
    return policy.compute_loss(batch)

def train_bc(train_dataloader, val_dataloader, config, policy_config):
    import time
    num_steps = config['num_steps']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    validate_every = config['validate_every']
    save_every = config['save_every']
    use_amp = config.get('bf16', False)
    max_validation_batches = config.get('max_validation_batches', 52)
    if max_validation_batches < 1:
        raise ValueError('max_validation_batches must be positive')

    set_seed(seed)
    torch.backends.cudnn.benchmark = config.get('cudnn_benchmark', True)
    policy = make_policy(policy_class, policy_config)
    if config['load_pretrain']:
        loading_status = policy.deserialize(
            torch.load(
                f'{config["pretrained_path"]}',
                map_location='cuda',
                weights_only=True,
            ),
            mode='transfer',
            transfer_source_embodiment=config.get('transfer_source_embodiment'),
        )
        print(f'loaded! {loading_status}')
    if config['resume_ckpt_path'] is not None:
        loading_status = policy.deserialize(
            torch.load(
                config['resume_ckpt_path'],
                map_location='cuda',
                weights_only=True,
            ),
            mode='strict',
        )
        print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
    resume_from_dir = config.get('resume_from_dir')
    resume_snapshot = None
    if resume_from_dir:
        snapshot_path = os.path.join(resume_from_dir, AUTO_RESUME_SNAPSHOT)
        if os.path.isfile(snapshot_path):
            resume_snapshot = torch.load(
                snapshot_path,
                map_location='cuda',
                weights_only=True,
            )
            required_keys = {'policy', 'optimizer', 'training_state'}
            missing_keys = required_keys - set(resume_snapshot)
            if missing_keys:
                raise RuntimeError(
                    f'Auto-resume snapshot {snapshot_path} is missing keys: '
                    f'{sorted(missing_keys)}'
                )
            model_dicts = resume_snapshot['policy']
            resume_policy_path = snapshot_path
        else:
            resume_policy_path = os.path.join(resume_from_dir, 'policy_last.ckpt')
            model_dicts = torch.load(
                resume_policy_path,
                map_location='cuda',
                weights_only=True,
            )
        loading_status = policy.deserialize(model_dicts, mode='strict')
        print(f'Resumed ALL model components from: {resume_policy_path}, Status: {loading_status}')
    policy.cuda()
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params}")
    if use_amp:
        print("Using BF16 mixed precision training")
    optimizer = make_optimizer(policy_class, policy)

    start_step = 0
    min_val_loss = np.inf
    if resume_from_dir:
        desired_lrs = [pg['lr'] for pg in optimizer.param_groups]
        if resume_snapshot is not None:
            optimizer.load_state_dict(resume_snapshot['optimizer'])
            training_state = resume_snapshot['training_state']
            print(f'Resumed optimizer from atomic snapshot in: {resume_from_dir}')
        else:
            resume_optimizer_path = os.path.join(resume_from_dir, 'optimizer_last.ckpt')
            if os.path.exists(resume_optimizer_path):
                optimizer.load_state_dict(torch.load(
                    resume_optimizer_path,
                    map_location='cuda',
                    weights_only=True,
                ))
                print(f'Resumed optimizer from: {resume_optimizer_path}')
            state_path = os.path.join(resume_from_dir, 'training_state.json')
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    training_state = json.load(f)
            else:
                training_state = None
                print(f'Warning: no training_state.json in {resume_from_dir}, starting from step 0')

        if resume_snapshot is not None or os.path.exists(
            os.path.join(resume_from_dir, 'optimizer_last.ckpt')
        ):
            for pg, lr in zip(optimizer.param_groups, desired_lrs):
                pg['lr'] = lr
            print(f'Applied LRs from script args: {desired_lrs}')
        if training_state is not None:
            start_step = training_state.get('step', 0) + 1
            min_val_loss = training_state.get('min_val_loss', float('inf'))
            print(f'Resuming from step {start_step}, min_val_loss: {min_val_loss}')

    best_ckpt_info = None
    if start_step > num_steps:
        print(
            f'Training already complete at step {start_step - 1}; '
            f'target num_steps is {num_steps}'
        )
        return None

    train_dataloader = repeater(train_dataloader)
    loss_list = []
    # l1_list = []
    # feature_loss_list = []
    val_loss_list = []
    tokenizer_params = list(policy.model.transformer_body.parameters())
    
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    # A crash may leave W&B history ahead of the last durable model checkpoint.
    # Keep its internal row index increasing while the explicit optimizer-update
    # axes describe the actual restored/replayed model state.
    wandb_log_offset = max(0, int(wandb.run.step) - start_step) if config['wandb'] else 0
    if wandb_log_offset:
        wandb.run.summary['resume_history_offset'] = wandb_log_offset
        wandb.run.summary['resumed_optimizer_updates'] = start_step

    for step in tqdm(range(start_step, num_steps+1)):
        iteration_start = time.monotonic()
        log_step = step + wandb_log_offset
        if step % validate_every == 0:
            print("training loss:", np.mean(loss_list) if len(loss_list) > 0 else "N/A")
            print('validating')

            with torch.inference_mode(), amp_ctx:
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    del forward_dict
                    if batch_idx + 1 >= max_validation_batches:
                        break

                validation_summary = compute_dict_mean(validation_dicts)

                epoch_val_loss = validation_summary['all_emb_loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = float(epoch_val_loss)
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
                    if config['wandb']:
                        wandb.run.summary['best_validation_loss'] = min_val_loss
                        wandb.run.summary['best_validation_optimizer_updates'] = step
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)     
            if config['wandb']:       
                wandb.log({**validation_summary, 'validation/optimizer_updates': step},
                          step=log_step, commit=False)
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        summary_string += f'{k}_{kk}: {vv:.3f} '
                else:
                    summary_string += f'{k}: {v:.3f} '
            print(summary_string)
            val_loss_list.append(epoch_val_loss)
            del validation_dicts, validation_summary, epoch_val_loss
                
        # training
        training_start = time.monotonic()
        policy.train() 
        optimizer.zero_grad(set_to_none=True)
        data = next(train_dataloader)
        with amp_ctx:
            forward_dict = forward_pass(data, policy)
            loss = forward_dict['all_emb_loss']
        loss.backward()
        loss_list.append(loss.item())
        
        optimizer.step()
        policy.post_step()
        telemetry = {
            'optimizer_updates': step + 1,
            'performance/train_step_seconds': time.monotonic() - training_start,
            'performance/gpu_peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'performance/gpu_peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
            **{f'learning_rate/group_{i}': group['lr']
               for i, group in enumerate(optimizer.param_groups)},
        }
        if config['wandb']:
            wandb.log({**forward_dict, **telemetry}, step=log_step, commit=False)
        
        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)
            #save optimizer state
            optimizer_ckpt_path = os.path.join(ckpt_dir, f'optimizer_step_{step}_seed_{seed}.ckpt')
            torch.save(optimizer.state_dict(), optimizer_ckpt_path)
            # save plots
            plot_path = os.path.join(ckpt_dir, f'train_val_seed_{seed}.png')
            plt.figure()
            plt.plot(np.linspace(0, len(loss_list), len(loss_list)), loss_list, label='train')
            plt.plot(np.linspace(0, len(loss_list), len(val_loss_list)), val_loss_list, label='validation')
            # plt.ylim([-0.1, 8.0])
            plt.tight_layout()
            plt.legend()
            plt.title('loss curve')
            plt.savefig(plot_path)
            plt.close('all')
            plt.clf()
            print(f'Saved plots to {ckpt_dir}')
        if step % 2000 == 0:
            _save_resume_checkpoint(policy, optimizer, ckpt_dir, step, min_val_loss)
            gc.collect()
            
        del forward_dict, loss, data
        telemetry['performance/iteration_seconds'] = time.monotonic() - iteration_start
        if config['wandb']:
            wandb.log(telemetry, step=log_step, commit=True)
            
    _save_resume_checkpoint(policy, optimizer, ckpt_dir, num_steps, min_val_loss)

    return best_ckpt_info

def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1

def _load_model_configs(args):
    with open(args['embodiment_config_path'], 'r') as f:
        emb_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(args['trunk_config_path'], 'r') as f:
        transformer_dict = yaml.load(f, Loader=yaml.FullLoader)
    return emb_dict, transformer_dict


def _apply_model_config_overrides(args, embodiment_config):
    for key in embodiment_config.keys():
        if key == 'human':
            for modality in embodiment_config[key]['action_head'].keys():
                head_config = embodiment_config[key]['action_head'][modality]
                if 'action_horizon' in head_config:
                    head_config['action_horizon'] = args['chunk_size']

    if args['gradient_checkpoint']:
        for key in embodiment_config.keys():
            if isinstance(embodiment_config[key], dict) and 'tokenizer' in embodiment_config[key]:
                for modality in embodiment_config[key]['tokenizer'].keys():
                    if 'image' in modality:
                        embodiment_config[key]['tokenizer'][modality]['gradient_checkpoint'] = True


def _validate_dream_targets(embodiment_config):
    """Require every configured dream target to be materializable by the loader."""
    for embodiment, config in embodiment_config.items():
        if not isinstance(config, dict) or not config.get('dream_head'):
            continue
        if embodiment != HTD_POLICY_EMBODIMENT:
            raise ValueError(
                f"HTD data loading does not support dream heads for embodiment "
                f"{embodiment!r}; expected {HTD_POLICY_EMBODIMENT!r}"
            )
        unsupported = sorted(
            set(config['dream_head']) - set(KEY_PATH_MAP_G1_INSPIRE_WB)
        )
        if unsupported:
            raise ValueError(
                "HTD dream targets are not supported by the current data loader: "
                f"{unsupported}"
            )


def _validate_loaded_dream_embodiments(embodiment_config, embodiment_list):
    active = [
        embodiment
        for embodiment in embodiment_list
        if isinstance(embodiment_config.get(embodiment), dict)
        and embodiment_config[embodiment].get('dream_head')
    ]
    if not active:
        raise ValueError(
            "HTD training data must include an embodiment with non-empty dream "
            f"heads; loaded embodiments: {list(embodiment_list)!r}"
        )


def _prepare_model_configs(args):
    model_configs = _load_model_configs(args)
    embodiment_config, transformer_config = model_configs
    if not isinstance(embodiment_config, dict) or not isinstance(transformer_config, dict):
        raise ValueError('model and trunk YAMLs must contain mappings')
    _apply_model_config_overrides(args, embodiment_config)
    has_dream_heads = any(isinstance(c, dict) and c.get('dream_head')
                          for c in embodiment_config.values())
    if args['touch_dreaming'] != bool(has_dream_heads):
        raise ValueError('--touch_dreaming requires non-empty dream heads; dream-head YAMLs require --touch_dreaming')
    if args['touch_dreaming'] and args['policy_class'] != 'MXT':
        raise ValueError('--touch_dreaming requires --policy_class MXT')
    for embodiment, config in embodiment_config.items():
        if not isinstance(config, dict):
            continue
        for group in ('action_head', 'dream_head'):
            heads = config.get(group, {})
            if not isinstance(heads, dict):
                raise ValueError(f'{embodiment}.{group} must be a mapping')
            for name, head in heads.items():
                if not isinstance(head, dict) or head.get('action_horizon') != args['chunk_size']:
                    raise ValueError(f'{embodiment}.{group}.{name} action_horizon must match chunk_size')
                if group == 'dream_head':
                    weight = head.get('loss_weight')
                    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight <= 0:
                        raise ValueError(f'{embodiment}.{group}.{name} loss_weight must be finite and positive')
    _validate_dream_targets(embodiment_config)

    resume_dir = args.get('resume_training')
    if resume_dir:
        path = os.path.join(resume_dir, 'all_configs.json')
        with open(path) as stream:
            previous = json.load(stream)
        if not isinstance(previous, dict) or previous.get('policy_class') != args['policy_class']:
            raise ValueError('resume policy_class does not match the current policy_class')
        saved = previous.get('policy_config')
        if not isinstance(saved, dict):
            raise ValueError('resume metadata is missing policy_config')
        for name, current in [('embodiment_args_dict', embodiment_config),
                              ('transformer_args', transformer_config)]:
            if saved.get(name) != current:
                raise ValueError(f'resume policy_config.{name} does not exactly match the current parsed configuration')
    return model_configs


def main_train(
    args,
    *,
    experiment_suffix=None,
    model_configs=None,
):
    set_seed(1)
    # command line parameters
    ckpt_dir = args['ckpt_dir']
    dataset_dir = args['dataset_dir']
    policy_class = args['policy_class']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_steps = args['num_steps']
    validate_every = args['validate_every']
    save_every = args['save_every']
    resume_ckpt_path = args['resume_ckpt_path']
    train_ratio = args['train_ratio']
    min_val_num = args['min_val_num']
    
    suffix = policy_class if experiment_suffix is None else experiment_suffix
    ckpt_dir = _effective_ckpt_dir(args, suffix)
    auto_resume_lock = None
    if os.path.isdir(ckpt_dir):
        if not args.get('auto_resume'):
            print(f'ckpt_dir {ckpt_dir} already exists, exiting')
            return
        print(f'Auto-resume will reuse ckpt_dir: {ckpt_dir}')
    if args.get('auto_resume'):
        os.makedirs(ckpt_dir, exist_ok=True)
        lock_path = os.path.join(ckpt_dir, '.auto-resume.lock')
        auto_resume_lock = open(lock_path, 'w')
        try:
            fcntl.flock(auto_resume_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            auto_resume_lock.close()
            raise RuntimeError(
                f'another auto-resume process is already using {ckpt_dir}'
            ) from exc
    args['ckpt_dir'] = ckpt_dir 
    # get task parameters
    is_sim = task_name[:4] == 'sim_'

    if model_configs is None:
        emb_dict, transformer_dict = _load_model_configs(args)
    else:
        emb_dict, transformer_dict = model_configs
    _apply_model_config_overrides(args, emb_dict)
    
    name_filter = lambda _: True

    print(f'===========================START===========================:')
    print(f"{task_name}")
    print(f'===========================Config===========================:')
    print(f'ckpt_dir: {ckpt_dir}')
    print(f'policy_class: {policy_class}')
    main_cam_h = args['main_cam_height'] if args['main_cam_height'] is not None else args['height']
    main_cam_w = args['main_cam_width'] if args['main_cam_width'] is not None else args['width']
    wrist_cam_h = args['wrist_cam_height'] if args['wrist_cam_height'] is not None else args['height']
    wrist_cam_w = args['wrist_cam_width'] if args['wrist_cam_width'] is not None else args['width']
    print(f'main_cam_size: ({main_cam_h}, {main_cam_w}), wrist_cam_size: ({wrist_cam_h}, {wrist_cam_w})')

    sample_weights = args['sample_weights']
    train_dataloader, val_dataloader, embodiment_stats, embodiment_list, _ = load_data(dataset_dir, name_filter, emb_dict, 
                                                                        batch_size_train, batch_size_val, args['chunk_size'], 
                                                                        args['skip_mirrored_data'],
                                                                        sample_weights=sample_weights,
                                                                        stats_dir_l=args.get('stats_dir'),
                                                                        num_workers=args.get('num_workers', 8),
                                                                        pin_memory=args.get('pin_memory', True),
                                                                        prefetch_factor=args.get('prefetch_factor', 1),
                                                                        persistent_workers=args.get('persistent_workers', True),
                                                                        train_ratio=train_ratio,
                                                                        main_cam_size=(main_cam_h, main_cam_w),
                                                                        wrist_cam_size=(wrist_cam_h, wrist_cam_w),
                                                                        data_aug=args['data_aug'],
                                                                        grayscale = args['grayscale'],
                                                                        randomize_color = args['randomize_color'],
                                                                        min_val_num=min_val_num,
                                                                        vel_rebalance=args['vel_rebalance'],
                                                                        vel_rebalance_weight_x=args['vel_rebalance_weight_x'],
                                                                        vel_rebalance_weight_y=args['vel_rebalance_weight_y'],
                                                                        vel_rebalance_weight_yaw=args['vel_rebalance_weight_yaw'],
                                                                        vel_rebalance_threshold=args['vel_rebalance_threshold'],
                                                            )
    if args['touch_dreaming']:
        _validate_loaded_dream_embodiments(emb_dict, embodiment_list)
    lr_backbone = 1e-5
        
    policy_config = {
        'lr': args['lr'],
        'lr_tokenizer': args['lr_tokenizer'],
        'lr_action_head': args['lr_action_head'],
        'lr_dream_head': args['lr_dream_head'],
        'lr_trunk': args['lr_trunk'],
        'lr_backbone': lr_backbone,
        'norm_stats': embodiment_stats,
        "embodiment_args_dict": emb_dict,
        "transformer_args": transformer_dict,
        "share_cross_attn": args['share_cross_attn'],
        }
    
    print(f'====================FINISH INIT POLICY========================:')
    config = {
        'num_steps': num_steps,
        'validate_every': validate_every,
        'save_every': save_every,
        'max_validation_batches': args.get('max_validation_batches', 52),
        'ckpt_dir': ckpt_dir,
        'resume_ckpt_path': resume_ckpt_path,
        'lr': args['lr'],
        'policy_class': policy_class,
        # 'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': False,
        'real_robot': not is_sim,
        'load_pretrain': args['load_pretrain'],
        'height':args['height'],
        'width':args['width'],
        'wandb': args['wandb'],
        'pretrained_path': args['pretrained_path'],
        'transfer_source_embodiment': args['transfer_source_embodiment'],
        'bf16': args['bf16'],
        'cudnn_benchmark': args.get('cudnn_benchmark', True),
        'resume_from_dir': args.get('resume_training'),
        'auto_resume': args.get('auto_resume', False),
        'touch_dreaming': args['touch_dreaming'],
    }
    all_configs = {**config, **args, 'policy_config': {k: v for k, v in policy_config.items() if k != 'norm_stats'}}
    print(all_configs)
    
    if args['width'] < 0:
        args['width'] = None
    if args['height'] < 0:
        args['height'] = None

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    config_path = os.path.join(ckpt_dir, 'config.pkl')
    all_config_path = os.path.join(ckpt_dir, 'all_configs.json')
    expr_name = ckpt_dir.split('/')[-1]
    if args['wandb']:
        wandb_kwargs = {}
        if args.get('auto_resume') and args.get('resume_training'):
            state_path = os.path.join(args['resume_training'], 'training_state.json')
            try:
                with open(state_path, 'r') as stream:
                    wandb_run_id = json.load(stream).get('wandb_run_id')
            except (OSError, UnicodeError, json.JSONDecodeError):
                wandb_run_id = None
            if wandb_run_id:
                wandb_kwargs.update(id=wandb_run_id, resume='allow')
        wandb.init(
            project=args['wandb_name'],
            reinit=True,
            name=expr_name,
            **wandb_kwargs,
        )
        wandb.config.update(all_configs, allow_val_change=True)
        wandb.define_metric('optimizer_updates')
        wandb.define_metric('*', step_metric='optimizer_updates')
        wandb.define_metric('validation/optimizer_updates')
        wandb.define_metric('val_*', step_metric='validation/optimizer_updates')
        wandb.define_metric('val_all_emb_loss', step_metric='validation/optimizer_updates', summary='min')
        _atomic_json_dump({
            'id': wandb.run.id, 'entity': wandb.run.entity,
            'project': wandb.run.project, 'url': wandb.run.url,
        }, os.path.join(ckpt_dir, 'wandb_run.json'), indent=2)
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    _atomic_json_dump(all_configs, all_config_path, indent=4)

    # save dataset stats
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(embodiment_stats, f)

    split = {}
    for name, loader in [('train', train_dataloader), ('validation', val_dataloader)]:
        dataset = loader.dataset
        split[name] = [
            {'path': os.fsdecode(dataset.dataset_path_list[int(index)]), 'frames': int(length)}
            for index, length in zip(dataset.episode_ids, dataset.episode_len)
        ]
    split['normalization_directories'] = args.get('stats_dir') or dataset_dir
    split_path = os.path.join(ckpt_dir, 'dataset_split.json')
    _atomic_json_dump(split, split_path, indent=2)
    if args['wandb']:
        wandb.save(all_config_path, base_path=ckpt_dir, policy='now')
        wandb.save(split_path, base_path=ckpt_dir, policy='now')

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config, policy_config)

    if best_ckpt_info is not None:
        best_step, min_val_loss, best_state_dict = best_ckpt_info
        ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
        _atomic_torch_save(best_state_dict, ckpt_path)
        print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')
    else:
        print('No best checkpoint recorded during training')
    if args['wandb']:
        wandb.run.summary['completed_optimizer_updates'] = num_steps + 1
        wandb.finish()
        


def build_parser(*, policy_class_default='MXT', argv=None):
    # Select the model YAML before the full parse; training defaults are shared.
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument('--touch_dreaming', action='store_true')
    selected, _ = selector.parse_known_args(argv)
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'htd')
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True, type=str, help='checkpoint output prefix')
    parser.add_argument('--touch_dreaming', action='store_true',
                        help='enable touch dreaming with the default 16D HTD recipe')
    parser.add_argument('--policy_class', default='MXT' if selected.touch_dreaming else policy_class_default,
                        action='store', type=str, help='policy_class, capitalize')
    parser.add_argument('--task_name', required=True, type=str, help='task name for saved artifacts')
    
    parser.add_argument('--batch_size', default=16, action='store', type=int, help='batch_size')
    parser.add_argument('--seed', default=6, action='store', type=int, help='seed')
    parser.add_argument('--num_steps', default=80000, action='store', type=int, help='num_steps')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers; 0 loads in the training process')
    parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=True,
                        help='Pin DataLoader memory for transfers to CUDA')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Batches prefetched per DataLoader worker')
    parser.add_argument('--persistent_workers', action=argparse.BooleanOptionalAction, default=False,
                        help='Keep DataLoader workers alive between iterations over the loader')
    parser.add_argument('--cudnn_benchmark', action=argparse.BooleanOptionalAction, default=True,
                        help='Benchmark convolution algorithms to select faster implementations')
    parser.add_argument('--max_validation_batches', type=int, default=52, help='Number of validation batches per evaluation')
    parser.add_argument('--stats_dir', nargs='+', default=None, help='Directories used for normalization statistics; pass training-only directories to exclude validation episodes')
    
    parser.add_argument('--lr', default=5e-5, action='store', type=float, help='lr')
    parser.add_argument('--lr_tokenizer', default=5e-5, action='store', type=float, help='lr_tokenizer')
    parser.add_argument('--lr_action_head', default=5e-5, action='store', type=float, help='lr_action_head')
    parser.add_argument('--lr_dream_head', default=5e-5, action='store', type=float, help='lr_dream_head')
    parser.add_argument('--lr_trunk', default=5e-5, action='store', type=float, help='lr_trunk')
    
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--pretrained_path', action='store', type=str, help='pretrained_path', required=False)
    parser.add_argument(
        '--transfer_source_embodiment',
        action='store',
        type=str,
        default=None,
        help='source embodiment for cross-attention transfer from a multi-embodiment checkpoint',
    )
    
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=10000, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')

    parser.add_argument('--chunk_size', default=60, action='store', type=int, help='chunk_size', required=False)
    
    # for image 
    parser.add_argument('--width', type=int, default=640, help="fallback width for all cameras (backward compat)")
    parser.add_argument('--height', type=int, default=480, help="fallback height for all cameras (backward compat)")
    parser.add_argument('--main_cam_width', type=int, default=1280, help="main camera width (overrides --width)")
    parser.add_argument('--main_cam_height', type=int, default=480, help="main camera height (overrides --height)")
    parser.add_argument('--wrist_cam_width', type=int, default=640, help="wrist camera width (overrides --width)")
    parser.add_argument('--wrist_cam_height', type=int, default=480, help="wrist camera height (overrides --height)")
    parser.add_argument('--data_aug', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--randomize_color', action='store_true')
    parser.add_argument('--gradient_checkpoint', action='store_true')
    parser.add_argument('--bf16', action=argparse.BooleanOptionalAction, default=True,
                        help='Use BF16 autocast for training and validation; --no-bf16 uses FP32')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    
    # dataset
    parser.add_argument('--dataset_dir', required=True, nargs='+', type=str, help='processed dataset folder(s); pass multiple dirs for multi-group training')
    parser.add_argument('--sample_weights', default=None, nargs='+', type=float, help='per-group sampling weights (one per dataset_dir); defaults to uniform')
    parser.add_argument('--train_ratio', default=0.99, action='store', type=float, help='train_ratio')
    parser.add_argument('--min_val_num', default=1, action='store', type=int, help='min_val_num')
    
    # mxt config
    parser.add_argument(
        '--embodiment_config_path',
        action='store',
        type=str,
        default=os.path.join(config_dir, 'dftp_nonwalk_latent16.yaml') if selected.touch_dreaming else None,
        required=not selected.touch_dreaming,
        help='embodiment YAML; required for MXT, optional override for HTD',
    )
    parser.add_argument(
        '--trunk_config_path',
        action='store',
        type=str,
        default=os.path.join(config_dir, 'trunk_nonwalk.yaml') if selected.touch_dreaming else None,
        required=not selected.touch_dreaming,
        help='path to the transformer trunk YAML paired with the embodiment config',
    )
    parser.add_argument('--share_cross_attn', action='store_true')
    
    parser.add_argument('--wandb_name', action='store', type=str, help='W&B project (default: htd)', default='htd')
        
    # velocity sampling rebalance
    parser.add_argument('--vel_rebalance', action='store_true', default=False, help='enable per-timestep velocity sampling rebalance')
    parser.add_argument('--vel_rebalance_weight_x', type=float, default=1.0, help='extra sampling weight for timesteps with active lin_vel_x')
    parser.add_argument('--vel_rebalance_weight_y', type=float, default=2.0, help='extra sampling weight for timesteps with active lin_vel_y')
    parser.add_argument('--vel_rebalance_weight_yaw', type=float, default=3.0, help='extra sampling weight for timesteps with active ang_vel_z')
    parser.add_argument('--vel_rebalance_threshold', type=float, default=0.1, help='velocity magnitude threshold to consider a component active')

    parser.add_argument('--resume_training', action='store', type=str, default=None, help='source ckpt_dir to resume from (loads model+optimizer+step; saves to new ckpt_dir)')
    parser.add_argument('--auto_resume', action='store_true', help='resume an existing computed ckpt_dir in place')

    return parser


def _validate_training_inputs(args, parser):
    """Validate split/group metadata before CUDA, statistics, or run artifacts."""
    for name in ('batch_size', 'validate_every', 'save_every', 'max_validation_batches',
                 'min_val_num', 'prefetch_factor'):
        if args[name] < 1:
            parser.error(f'--{name} must be positive')
    if args['chunk_size'] is None or args['chunk_size'] < 1:
        parser.error('--chunk_size must be supplied and positive')
    if args['num_steps'] < 0:
        parser.error('--num_steps must be nonnegative')
    if args['num_workers'] < 0:
        parser.error('--num_workers must be nonnegative')
    if not 0 < args['train_ratio'] <= 1:
        parser.error('--train_ratio must be in (0, 1]')
    directories = args['dataset_dir']
    counts = {}

    def episode_count(directory, label):
        if not os.path.isdir(directory):
            parser.error(f'{label} directory not found: {directory}')
        if directory not in counts:
            counts[directory] = len(find_all_hdf5(directory, args['skip_mirrored_data']))
        return counts[directory]

    group_counts = [episode_count(directory, 'dataset') for directory in directories]
    for directory, count in zip(directories, group_counts):
        if not count:
            parser.error(f'no HDF5 episodes found in dataset directory: {directory}')
    validation_count = max(args['min_val_num'], int(group_counts[0] * (1 - args['train_ratio'])))
    if validation_count >= group_counts[0]:
        parser.error(
            f'first dataset group has {group_counts[0]} episodes but the split reserves '
            f'{validation_count} for validation, leaving that training group empty; '
            'add episodes or reduce --min_val_num'
        )
    weights = args.get('sample_weights')
    if weights is not None:
        if (len(weights) != len(directories) or not np.isfinite(weights).all()
                or any(weight < 0 for weight in weights) or sum(weights) <= 0):
            parser.error('--sample_weights must provide one finite nonnegative weight per dataset directory, with positive total')
    stats_directories = args.get('stats_dir')
    if stats_directories is not None:
        if not sum(episode_count(directory, 'normalization') for directory in stats_directories):
            parser.error('no HDF5 episodes found in normalization directories')


def main(argv=None, *, parser=None):
    if parser is None:
        parser = build_parser(argv=argv)
    args = vars(parser.parse_args(argv))

    _validate_training_inputs(args, parser)

    experiment_suffix = 'HTD' if args['touch_dreaming'] else None
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    _configure_auto_resume(args, experiment_suffix)
    try:
        model_configs = _prepare_model_configs(args)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    torch.cuda.set_device(args['gpu_id'])
    return main_train(
        args,
        experiment_suffix=experiment_suffix,
        model_configs=model_configs,
    )


if __name__ == '__main__':
    main()
