"""Train ACT or decoder-only ACT (HIT)."""

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import numpy as np
import os
import pickle
import argparse
import contextlib
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
import json
import yaml
import wandb
from act_utils import compute_dict_mean, find_all_hdf5, set_seed, load_data
from policy import make_policy, make_optimizer

def forward_pass(data, policy):
    image_data, *state_data = data

    def prepare_image(image):
        image = image.cuda(non_blocking=True)
        if image.dtype == torch.uint8:
            image = image.float()
            # A tensor divisor preserves the CPU FP32 normalization on CUDA.
            image.div_(image.new_tensor(255.0))
        return image

    image_data = ({name: prepare_image(image) for name, image in image_data.items()}
                  if isinstance(image_data, dict) else prepare_image(image_data))
    qpos_data, action_data, is_pad, action_mask_data = (
        tensor.cuda(non_blocking=True) for tensor in state_data
    )
    return policy(qpos_data, image_data, action_data, is_pad, action_mask_data)

def train_bc(train_dataloader, val_dataloader, config):
    num_steps = config['num_steps']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    validate_every = config['validate_every']
    save_every = config['save_every']
    use_amp = config.get('bf16', False)
    max_validation_batches = config.get('max_validation_batches', 52)
    if max_validation_batches < 1:
        raise ValueError('max_validation_batches must be positive')

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    if config['load_pretrain']:
        loading_status = policy.deserialize(torch.load(f'{config["pretrained_path"]}/policy_last.ckpt', map_location='cuda'))
        print(f'loaded! {loading_status}')
    if config['resume_ckpt_path'] is not None:
        loading_status = policy.deserialize(torch.load(config['resume_ckpt_path']))
        print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)
    if config['load_pretrain']:
        optimizer.load_state_dict(torch.load(f'{config["pretrained_path"]}/optimizer_last.ckpt', map_location='cuda'))
        

    min_val_loss = np.inf
    best_ckpt_info = None
    
    train_dataloader = repeater(train_dataloader)
    loss_list = []
    l1_list = []
    if policy_class == 'HIT':
        feature_loss_list = []
    val_loss_list = []
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    for step in tqdm(range(num_steps+1)):
        if step % validate_every == 0:
            print("training loss:", np.mean(loss_list) if len(loss_list) > 0 else "N/A")
            print("training l1 loss:", np.mean(l1_list) if len(l1_list) > 0 else "N/A")
            if policy_class == 'HIT':
                print("training feature loss:", np.mean(feature_loss_list) if len(feature_loss_list) > 0 else "N/A")
            l1_list = []
            if policy_class == 'HIT':
                feature_loss_list = []
            print('validating')

            with torch.inference_mode(), amp_ctx:
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    if batch_idx + 1 >= max_validation_batches:
                        break

                validation_summary = compute_dict_mean(validation_dicts)

                epoch_val_loss = validation_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)     
            if config['wandb']:       
                wandb.log(validation_summary, step=step)
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)
            val_loss_list.append(epoch_val_loss.item())
                
        # training
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        data = next(train_dataloader)
        with amp_ctx:
            forward_dict = forward_pass(data, policy)
            loss = forward_dict['loss']
        loss.backward()
        loss_list.append(loss.item())
        l1_list.append(forward_dict['l1'].item())
        if policy_class == 'HIT' and 'feature_loss' in forward_dict:
            feature_loss_list.append(forward_dict['feature_loss'].item())
        optimizer.step()
        if config['wandb']:
            wandb.log(forward_dict, step=step)

        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)
            # Save optimizer state.
            optimizer_ckpt_path = os.path.join(ckpt_dir, f'optimizer_step_{step}_seed_{seed}.ckpt')
            torch.save(optimizer.state_dict(), optimizer_ckpt_path)
            # save plots
            plot_path = os.path.join(ckpt_dir, f'train_val_seed_{seed}.png')
            plt.figure()
            plt.plot(np.linspace(0, len(loss_list), len(loss_list)), loss_list, label='train')
            plt.plot(np.linspace(0, len(loss_list), len(val_loss_list)), val_loss_list, label='validation')
            plt.ylim([-0.1, 8.0])
            plt.tight_layout()
            plt.legend()
            plt.title('loss curve')
            plt.savefig(plot_path)
            plt.close()
            print(f'Saved plots to {ckpt_dir}')
        if step % 2000 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
            torch.save(policy.serialize(), ckpt_path)
            optimizer_ckpt_path = os.path.join(ckpt_dir, f'optimizer_last.ckpt')
            torch.save(optimizer.state_dict(), optimizer_ckpt_path)
            
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)
    optimizer_ckpt_path = os.path.join(ckpt_dir, f'optimizer_last.ckpt')
    torch.save(optimizer.state_dict(), optimizer_ckpt_path)

    return best_ckpt_info

def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1

import math

def _resnet_feat_size(dim):
    """Compute ResNet18 output spatial size for a given input dimension."""
    x = (dim - 1) // 2 + 1          # conv1: 7x7, stride 2, pad 3
    x = (x - 1) // 2 + 1            # maxpool: 3x3, stride 2, pad 1
    x = math.ceil(x / 2)            # layer2 stride
    x = math.ceil(x / 2)            # layer3 stride
    x = math.ceil(x / 2)            # layer4 stride
    return x

def _compute_num_image_tokens(camera_names, height, width, wrist_height=None, wrist_width=None):
    """Count spatial tokens using the size of each camera view."""
    has_split_main = any(c in ('main_left', 'main_right') for c in camera_names)
    h_feat = _resnet_feat_size(height)
    if has_split_main:
        w_feat = _resnet_feat_size(width // 2)
    else:
        w_feat = _resnet_feat_size(width)
    return sum(
        _resnet_feat_size(wrist_height) * _resnet_feat_size(wrist_width)
        if 'wrist' in name and wrist_width is not None else h_feat * w_feat
        for name in camera_names
    )

def main_train(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_steps = args['num_steps']
    eval_every = args['eval_every']
    validate_every = args['validate_every']
    save_every = args['save_every']
    resume_ckpt_path = args['resume_ckpt_path']
    backbone = args['backbone']
    same_backbones = args['same_backbones']
    dataset_dir = args['dataset_dir']
    train_ratio = args['train_ratio']
    min_val_num = args['min_val_num']
    
    precomputed_features = args['precomputed_features']
    
    ckpt_dir = f'{ckpt_dir}_{task_name}_{policy_class}'
    if os.path.isdir(ckpt_dir):
        print(f'ckpt_dir {ckpt_dir} already exists, exiting')
        return
    args['ckpt_dir'] = ckpt_dir 
    # get task parameters
    data_source = args.get('data_source', 'auto')
    if data_source == 'auto':
        is_sim = task_name.startswith('sim_')
        print(
            f"Data source inferred as {'sim' if is_sim else 'real'} from task_name "
            f"'{task_name}'. Pass --data_source explicitly to avoid name-based inference."
        )
    else:
        is_sim = data_source == 'sim'
    print(f'data_source: {"sim" if is_sim else "real"}')
    
    name_filter = lambda _: True

    # Load observation, action, and camera fields from YAML.
    if args.get('config') is not None:
        with open(args['config'], 'r') as f:
            data_config = yaml.safe_load(f)
        observation_name = data_config['observation_name']
        action_name = data_config['action_name']
        camera_names = data_config['camera_names']
        backbone_map = data_config.get('backbone_map', None)
        tactile_config = data_config.get('tactile_config', None)
    else:
        backbone_map = None
        tactile_config = None
        observation_name = ['proprioceptions/body',
                            'proprioceptions/eef',
                            'proprioceptions/gripper',
                            'proprioceptions/eef_to_body']
        action_name = ['actions/body', 'actions/eef', 'actions/gripper']
        camera_names = ['main_left', 'main_right', 'wrist']

    has_split_main = any(name in ('main_left', 'main_right') for name in camera_names)
    if has_split_main and args['width'] % 2:
        raise ValueError('Combined stereo head width must be even')
    if args.get('wrist_cam_width') is None:
        args['wrist_cam_width'] = args['width'] // 2 if has_split_main else args['width']
        args['wrist_cam_height'] = args['height']

    print(f'===========================START===========================:')
    print(f"{task_name}")
    print(f'===========================Config===========================:')
    print(f'ckpt_dir: {ckpt_dir}')
    print(f'policy_class: {policy_class}')
    print(f'observation_name: {observation_name}')
    print(f'action_name: {action_name}')
    print(f'camera_names: {camera_names}')
    train_dataloader, val_dataloader, stats, _, qpos_dim, action_dim = load_data(dataset_dir, name_filter, camera_names, 
                                                                                batch_size_train, batch_size_val, args['chunk_size'], 
                                                                                args['skip_mirrored_data'], args['load_pretrain'], 
                                                                                policy_class,
                                                                                train_ratio=train_ratio,
                                                                                stats_dir_l=args.get('stats_dir'),
                                                                                num_workers=args.get('num_workers', 8),
                                                                                pin_memory=args.get('pin_memory', True),
                                                                                width=args['width'],
                                                                                height=args['height'],
                                                                                main_cam_size=(args['height'], args['width']),
                                                                                wrist_cam_size=(args['wrist_cam_height'], args['wrist_cam_width']),
                                                                                normalize_resnet=args['normalize_resnet'],
                                                                                data_aug=args['data_aug'],
                                                                                observation_name=observation_name,
                                                                                feature_loss = args['feature_loss_weight'] > 0,
                                                                                grayscale = args['grayscale'],
                                                                                randomize_color = args['randomize_color'],
                                                                                randomize_data_degree = args['randomize_data_degree'],
                                                                                randomize_data = args['randomize_data'],
                                                                                action_name=action_name,
                                                                                precomputed_features = precomputed_features,
                                                                                min_val_num=min_val_num,
                                                                                is_sim=is_sim,
                                                                                )
    state_dim = qpos_dim
    if tactile_config:
        tactile_total = tactile_config['num_hands'] * tactile_config['tactile_dim']
        hit_proprio_dim = state_dim - tactile_total
        stats['qpos_mean'][hit_proprio_dim:] = 0.0
        stats['qpos_std'][hit_proprio_dim:] = 1.0
        print(f'HIT: skipping z-norm for tactile dims [{hit_proprio_dim}:{state_dim}] '
              f'(proprio_dim={hit_proprio_dim}, tactile_total={tactile_total})')
    state_mask = np.ones(state_dim).tolist()
    action_mask = np.ones(action_dim).tolist()
    if args['use_mask']:
        state_dim = sum(state_mask)
        action_dim = sum(action_mask)
        state_idx = np.where(state_mask)[0].tolist()
        action_idx = np.where(action_mask)[0].tolist()
    else:
        state_idx = np.arange(state_dim).tolist()
        action_idx = np.arange(action_dim).tolist()
    lr_backbone = 1e-5
    backbone = args['backbone']
    if policy_class == 'ACT':
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'same_backbones': args['same_backbones'],
                         'enc_layers': args['enc_layers'],
                         'dec_layers': args['dec_layers'],
                         'nheads': args['nheads'],
                         'camera_names': camera_names,
                         'vq': False,
                         'action_dim': action_dim,
                         'state_dim': state_dim,
                         'no_vae_encoder': args['no_vae_encoder'],
                         'state_idx': state_idx,
                         'action_idx': action_idx,
                         'state_mask': state_mask,
                         'action_mask': action_mask,
                         'precomputed_features': precomputed_features,
                         'backbone_map': backbone_map,
                         }
    elif policy_class == 'HIT':
        num_image_tokens = _compute_num_image_tokens(
            camera_names, args['height'], args['width'],
            args['wrist_cam_height'], args['wrist_cam_width'],
        )
        num_tactile_tokens = tactile_config['num_hands'] if tactile_config else 0
        if tactile_config:
            tactile_total = tactile_config['num_hands'] * tactile_config['tactile_dim']
            proprio_dim = state_dim - tactile_total
        else:
            proprio_dim = state_dim
        context_len = 1 + num_image_tokens + num_tactile_tokens + args['chunk_size']
        print(f'HIT context_len: {context_len} (1 proprio + {num_image_tokens} image + {num_tactile_tokens} tactile + {args["chunk_size"]} action)')
        policy_config = {'lr': args['lr'],
                         'hidden_dim': args['hidden_dim'],
                         'dec_layers': args['dec_layers'],
                         'nheads': args['nheads'],
                         'num_queries': args['chunk_size'],
                         'camera_names': camera_names,
                         'action_dim': action_dim,
                         'state_dim': state_dim,
                         'backbone': backbone,
                         'same_backbones': args['same_backbones'],
                         'lr_backbone': lr_backbone,
                         'context_len': context_len,
                         'num_queries': args['chunk_size'], 
                         'use_pos_embd_image': args['use_pos_embd_image'],
                         'use_pos_embd_action': args['use_pos_embd_action'],
                         'feature_loss': args['feature_loss_weight']>0,
                         'feature_loss_weight': args['feature_loss_weight'],
                         'self_attention': args['self_attention']==1,
                         'state_idx': state_idx,
                         'action_idx': action_idx,
                         'state_mask': state_mask,
                         'action_mask': action_mask,
                         'width': args['width'],
                         'height': args['height'],
                         'precomputed_features': precomputed_features,
                         'backbone_map': backbone_map,
                         'tactile_config': tactile_config,
                         'proprio_dim': proprio_dim,
                         }
    else:
        raise NotImplementedError
    print(f'====================FINISH INIT POLICY========================:')
    config = {
        'num_steps': num_steps,
        'eval_every': eval_every,
        'validate_every': validate_every,
        'save_every': save_every,
        'max_validation_batches': args.get('max_validation_batches', 52),
        'ckpt_dir': ckpt_dir,
        'resume_ckpt_path': resume_ckpt_path,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': False,
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'load_pretrain': args['load_pretrain'],
        'height':args['height'],
        'width':args['width'],
        'wrist_cam_width': args['wrist_cam_width'],
        'wrist_cam_height': args['wrist_cam_height'],
        'normalize_resnet': args['normalize_resnet'],
        'wandb': args['wandb'],
        'bf16': args['bf16'],
        'pretrained_path': args['pretrained_path'],
        'randomize_data_degree': args['randomize_data_degree'],
        'randomize_data': args['randomize_data'],
    }
    all_configs = {**config, **args}
    all_configs['resolved_data_config'] = {
        'observation_name': observation_name, 'action_name': action_name,
        'camera_names': camera_names, 'backbone_map': backbone_map,
        'tactile_config': tactile_config,
    }
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
    if not is_eval and args['wandb']:
        wandb.init(project=PROJECT_NAME, reinit=True, name=expr_name)
        wandb.config.update(config)
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    with open(all_config_path, 'w') as fp:
        json.dump(all_configs, fp, indent=4)

    # save dataset stats
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')
    if args['wandb']:
        wandb.finish()
        


def _resolve_act_variant(args, parser):
    """Select ACT or HIT from --decoder_only."""
    decoder_only = args.pop('decoder_only')

    if decoder_only and args['no_vae_encoder']:
        parser.error(
            '--no_vae_encoder only disables the standard ACT CVAE encoder; '
            'it cannot be combined with --decoder_only'
        )

    args['act_variant'] = 'decoder_only' if decoder_only else 'standard'
    args['policy_class'] = 'HIT' if decoder_only else 'ACT'
    args['model_type'] = args['policy_class']
    return args


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str,
                        help='YAML defining G1 observation, action, and camera fields')
    parser.add_argument('--eval', action='store_true',
                        help='evaluation is not implemented by this trainer; decoder-only ACT robot inference is in policy_rollout/')
    parser.add_argument('--onscreen_render', action='store_true',
                        help='legacy compatibility flag; ignored by ACT training')
    parser.add_argument('--ckpt_dir', required=True, type=str, help='checkpoint output prefix')
    parser.add_argument(
        '--decoder_only',
        action='store_true',
        help='Use non-autoregressive decoder-only ACT (HIT).',
    )
    parser.add_argument('--task_name', required=True, type=str, help='task name for saved artifacts')
    parser.add_argument(
        '--data_source',
        choices=['auto', 'real', 'sim'],
        default='auto',
        help="Action alignment source. 'auto' infers from a 'sim_' task-name prefix.",
    )
    
    parser.add_argument('--batch_size', default=16, action='store', type=int, help='batch_size')
    parser.add_argument('--seed', default=6, action='store', type=int, help='seed')
    parser.add_argument('--num_steps', default=80000, action='store', type=int, help='num_steps')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers; 0 loads in the training process')
    parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=True,
                        help='Pin DataLoader memory for transfers to CUDA')
    parser.add_argument('--max_validation_batches', type=int, default=52, help='Number of validation batches per evaluation')
    parser.add_argument('--stats_dir', default=None, help='Directory used for normalization statistics; pass the training-only directory to exclude validation episodes')
    parser.add_argument('--lr', default=2e-5, action='store', type=float, help='lr')
    parser.add_argument('--bf16', action=argparse.BooleanOptionalAction, default=True,
                        help='Use BF16 autocast for training and validation; parameters remain FP32')
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--pretrained_path', action='store', type=str, help='pretrained_path', required=False)
    
    parser.add_argument('--eval_every', action='store', type=int, default=100000,
                        help='legacy compatibility option; use --validate_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=10000, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=float, help='KL Weight', default=0.1)
    parser.add_argument('--enc_layers', action='store', type=int, default=4, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=2048)
    parser.add_argument(
        '--no_vae_encoder',
        action='store_true',
        help='Disable the standard ACT CVAE action encoder; this does not select decoder-only ACT.',
    )
    parser.add_argument(
        '--no_encoder',
        dest='no_vae_encoder',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    #dec_layers
    parser.add_argument('--dec_layers', action='store', type=int, default=8, required=False)
    parser.add_argument('--nheads', action='store', type=int, default=8, required=False)
    parser.add_argument('--use_pos_embd_image', action='store', type=int, default=1, required=False)
    parser.add_argument('--use_pos_embd_action', action='store', type=int, default=1, required=False)
    
    # dataset
    parser.add_argument('--dataset_dir', required=True, type=str, help='processed dataset directory')
    parser.add_argument('--train_ratio', action='store', type=float, default=0.99, required=False)
    parser.add_argument('--min_val_num', action='store', type=int, default=1)
    
    #feature_loss_weight
    parser.add_argument('--feature_loss_weight', action='store', type=float, default=0.0,
                        help='visual feature loss weight for decoder-only ACT')
    #self_attention
    parser.add_argument('--self_attention', action="store", type=int, default=1)
    
    #for backbone 
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--same_backbones', action='store_true')
    #use mask
    parser.add_argument('--use_mask', action='store_true')
    
    parser.add_argument('--precomputed_features', action='store_true')
    
    # for image 
    parser.add_argument('--width', '--main_cam_width', type=int, default=1280,
                        help='combined stereo head width before splitting')
    parser.add_argument('--height', '--main_cam_height', type=int, default=480,
                        help='head image height')
    parser.add_argument('--wrist_cam_width', type=int, default=None,
                        help='wrist image width; defaults to the width of one head view')
    parser.add_argument('--wrist_cam_height', type=int, default=None,
                        help='wrist image height; defaults to the head image height')
    parser.add_argument('--data_aug', action='store_true')
    parser.add_argument('--normalize_resnet', action='store_true') ### not used - always normalize - in the model.forward
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--randomize_color', action='store_true')
    parser.add_argument('--randomize_data', action='store_true')
    parser.add_argument('--randomize_data_degree', action='store', type=int, default=3)
    
    parser.add_argument('--wandb', action='store_true')
    
    parser.add_argument('--gpu_id', type=int, default=0)
    
    parser.add_argument('--wandb_name', action='store', type=str, help='W&B project (default: htd)', default='htd')
    
    return parser


def _validate_training_inputs(args, parser):
    """Check CLI intent and split sizes without opening data or initializing CUDA."""
    if args['eval']:
        parser.error('--eval is not supported by this trainer; decoder-only ACT robot inference is in policy_rollout/')
    for name in ('batch_size', 'validate_every', 'save_every', 'max_validation_batches', 'min_val_num'):
        if args[name] < 1:
            parser.error(f'--{name} must be positive')
    if args['num_steps'] < 0:
        parser.error('--num_steps must be nonnegative')
    if args['num_workers'] < 0:
        parser.error('--num_workers must be nonnegative')
    for name in ('width', 'height'):
        if args[name] < 1:
            parser.error(f'--{name} must be positive')
    wrist_sizes = (args.get('wrist_cam_width'), args.get('wrist_cam_height'))
    if any(value is not None for value in wrist_sizes):
        if any(value is None or value < 1 for value in wrist_sizes):
            parser.error('Supply positive --wrist_cam_width and --wrist_cam_height together')
    for name in ('hidden_dim', 'chunk_size'):
        if args[name] is None or args[name] < 1:
            parser.error(f'--{name} must be supplied and positive')
    if not 0 < args['train_ratio'] <= 1:
        parser.error('--train_ratio must be in (0, 1]')
    if not os.path.isfile(args['config']):
        parser.error(f'data configuration file not found: {args["config"]}')
    dataset_dir = args['dataset_dir']
    if not os.path.isdir(dataset_dir):
        parser.error(f'dataset directory not found: {dataset_dir}')
    episodes = find_all_hdf5(dataset_dir, args['skip_mirrored_data'])
    if not episodes:
        parser.error(f'no HDF5 episodes found in dataset directory: {dataset_dir}')
    validation_count = max(args['min_val_num'], int(len(episodes) * (1 - args['train_ratio'])))
    if validation_count >= len(episodes):
        parser.error(
            f'dataset has {len(episodes)} episodes but the split reserves '
            f'{validation_count} for validation, leaving no training episodes; '
            'add episodes or reduce --min_val_num'
        )
    stats_dir = args.get('stats_dir')
    if stats_dir is not None:
        if not os.path.isdir(stats_dir):
            parser.error(f'normalization directory not found: {stats_dir}')
        if not find_all_hdf5(stats_dir, args['skip_mirrored_data']):
            parser.error(f'no HDF5 episodes found in normalization directory: {stats_dir}')


def main(argv=None):
    global PROJECT_NAME
    parser = build_parser()
    args = _resolve_act_variant(vars(parser.parse_args(argv)), parser)
    _validate_training_inputs(args, parser)
    torch.cuda.set_device(args['gpu_id'])
    if args['bf16'] and not torch.cuda.is_bf16_supported():
        parser.error('Selected CUDA device does not support BF16; pass --no-bf16 to use FP32')
    PROJECT_NAME = args['wandb_name']
    main_train(args)


if __name__ == '__main__':
    main()
