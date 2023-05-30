import math
import random
from collections import defaultdict

import torch
import numpy as np
from functools import reduce

from khrylib.utils.torch import LongTensor
import torch.nn as nn
from khrylib.rl.core.distributions import Categorical, DiagGaussian
from khrylib.rl.core.policy import Policy
from khrylib.rl.core.running_norm import RunningNorm
from khrylib.models.mlp import MLP
from khrylib.utils.math import *
from design_opt.utils.tools import *
from design_opt.models.gnn import GNNSimple
from design_opt.models.jsmlp import JSMLP, ObsEncoder
from design_opt.models.group_action import LearnedGroup

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transform2ActPolicy(Policy):
    def __init__(self, cfg, agent):
        super().__init__()
        self.type = 'gaussian'
        self.cfg = cfg
        self.agent = agent
        self.enable_infer = agent.cfg.enable_infer
        self.attr_fixed_dim = agent.attr_fixed_dim
        self.sim_obs_dim = agent.sim_obs_dim
        self.sim_obs_hfield_dim = agent.sim_obs_hfield_dim
        self.sim_obs_obj_dim = agent.sim_obs_obj_dim
        self.sim_obs_goal_dim = agent.sim_obs_goal_dim
        self.attr_design_dim = agent.attr_design_dim
        self.control_state_dim = self.attr_fixed_dim + self.sim_obs_dim + self.attr_design_dim
        self.sym_embed_dim = sum(agent.cfg.num_legs) * 2
        if self.enable_infer:
            self.skel_state_dim = self.attr_fixed_dim + self.attr_design_dim + self.sym_embed_dim
            self.attr_state_dim = self.attr_fixed_dim + self.attr_design_dim + self.sym_embed_dim
        else:
            self.skel_state_dim = self.attr_fixed_dim + self.attr_design_dim
            self.attr_state_dim = self.attr_fixed_dim + self.attr_design_dim
        self.skel_action_dim = agent.skel_num_action
        self.control_action_dim = agent.control_action_dim
        self.attr_action_dim = agent.attr_design_dim
        self.action_dim = self.control_action_dim + self.attr_action_dim + 1
        self.skel_uniform_prob = cfg.get('skel_uniform_prob', 0.0)

        self.shared_std_att = 1
        self.shared_std_ctl = 1
        if self.enable_infer:
            self.group = LearnedGroup(ns=agent.cfg.num_legs,
                                      struct_subgroup=agent.cfg.struct_subgroup,
                                      updating_subgroup=agent.cfg.updating_subgroup,
                                      init_subgroup_name = agent.cfg.init_subgroup_name,
                                      subgroup_alpha_gap = agent.cfg.subgroup_alpha_gap) # [3,4,5,6]
            self.index_base = self.agent.env.index_base
            self.max_index = 256 #cfg['control_index_mlp'].get('max_index', 256)
            self.is_new_morphology = False

        # shared obs
        self.hfield_enc = ObsEncoder(self.sim_obs_hfield_dim)
        self.obj_enc = ObsEncoder(self.sim_obs_obj_dim)
        self.goal_enc = ObsEncoder(self.sim_obs_goal_dim)
        shared_obs_dim = self.hfield_enc.out_dim + self.obj_enc.out_dim + self.goal_enc.out_dim

        # skeleton transform
        self.skel_norm = RunningNorm(self.attr_state_dim)
        cur_dim = self.attr_state_dim
        if 'skel_pre_mlp' in cfg:
            self.skel_pre_mlp = MLP(cur_dim, cfg['skel_pre_mlp'], cfg['htype'])
            cur_dim = self.skel_pre_mlp.out_dim
        else:
            self.skel_pre_mlp = None
        if 'skel_gnn_specs' in cfg:
            self.skel_gnn = GNNSimple(cur_dim, cfg['skel_gnn_specs'])
            cur_dim = self.skel_gnn.out_dim
        else:
            self.skel_gnn = None
        if 'skel_mlp' in cfg:
            self.skel_mlp = MLP(cur_dim, cfg['skel_mlp'], cfg['htype'])
            cur_dim = self.skel_mlp.out_dim
        else:
            self.skel_mlp = None

        cur_dim += shared_obs_dim
        if 'skel_index_mlp' in cfg:
            imlp_cfg = cfg['skel_index_mlp']
            self.skel_ind_mlp = JSMLP(cur_dim, imlp_cfg['hdims'], self.skel_action_dim,
                                      imlp_cfg.get('max_index', 256),
                                      imlp_cfg.get('htype', 'tanh'), imlp_cfg['rescale_linear'],
                                      imlp_cfg.get('zero_init', False))
        else:
            self.skel_ind_mlp = None
            self.skel_action_logits = nn.Linear(cur_dim, self.skel_action_dim)

        # attribute transform
        self.attr_norm = RunningNorm(self.skel_state_dim) if cfg.get('attr_norm', True) else None
        cur_dim = self.skel_state_dim
        if 'attr_pre_mlp' in cfg:
            self.attr_pre_mlp = MLP(self.skel_state_dim, cfg['attr_pre_mlp'], cfg['htype'])
            cur_dim = self.attr_pre_mlp.out_dim
        else:
            self.attr_pre_mlp = None
        if 'attr_gnn_specs' in cfg:
            self.attr_gnn = GNNSimple(cur_dim, cfg['attr_gnn_specs'])
            cur_dim = self.attr_gnn.out_dim
        else:
            self.attr_gnn = None
        if 'attr_mlp' in cfg:
            self.attr_mlp = MLP(cur_dim, cfg['attr_mlp'], cfg['htype'])
            cur_dim = self.attr_mlp.out_dim
        else:
            self.attr_mlp = None

        cur_dim += shared_obs_dim
        if 'attr_index_mlp' in cfg:
            imlp_cfg = cfg['attr_index_mlp']
            self.attr_ind_mlp = JSMLP(cur_dim, imlp_cfg['hdims'],
                                      self.attr_action_dim * self.shared_std_att, imlp_cfg.get('max_index', 256),
                                      imlp_cfg.get('htype', 'tanh'), imlp_cfg['rescale_linear'],
                                      imlp_cfg.get('zero_init', False))
        else:
            self.attr_ind_mlp = None
            self.attr_action_mean = nn.Linear(cur_dim, self.attr_action_dim * self.shared_std_att)
            init_fc_weights(self.attr_action_mean)

        attr_action_log_std = torch.ones(1, self.attr_action_dim) * cfg['attr_log_std']
        self.attr_action_log_std = nn.Parameter(attr_action_log_std, requires_grad = not cfg['fix_attr_std'])

        # execution
        self.control_norm = RunningNorm(self.control_state_dim)
        cur_dim = self.control_state_dim
        if 'control_pre_mlp' in cfg:
            self.control_pre_mlp = MLP(cur_dim, cfg['control_pre_mlp'], cfg['htype'])
            cur_dim = self.control_pre_mlp.out_dim
        else:
            self.control_pre_mlp = None
        if 'control_gnn_specs' in cfg:
            self.control_gnn = GNNSimple(cur_dim, cfg['control_gnn_specs'])
            cur_dim = self.control_gnn.out_dim
        else:
            self.control_gnn = None
        if 'control_mlp' in cfg:
            self.control_mlp = MLP(cur_dim, cfg['control_mlp'], cfg['htype'])
            cur_dim = self.control_mlp.out_dim
        else:
            self.control_mlp = None

        cur_dim += shared_obs_dim
        if 'control_index_mlp' in cfg:
            imlp_cfg = cfg['control_index_mlp']
            self.control_ind_mlp = JSMLP(cur_dim, imlp_cfg['hdims'],
                                         self.control_action_dim * self.shared_std_ctl, imlp_cfg.get('max_index', 256),
                                         imlp_cfg.get('htype', 'tanh'), imlp_cfg['rescale_linear'],
                                         imlp_cfg.get('zero_init', False))
        else:
            self.control_ind_mlp = None
            self.control_action_mean = nn.Linear(cur_dim, self.control_action_dim * self.shared_std_ctl)
            init_fc_weights(self.control_action_mean)
        self.control_action_log_std = nn.Parameter(torch.ones(1, self.control_action_dim) * cfg['control_log_std'], requires_grad=not cfg['fix_control_std'])

    def batch_data(self, x):
        if len(x[0]) == 8:
            obs, edges, use_transform_action, num_nodes, body_ind, hfield, obj, goal = zip(*x)
        else:
            raise NotImplementedError
        obs = torch.cat(obs)
        use_transform_action = np.concatenate(use_transform_action)
        num_nodes = np.concatenate(num_nodes)
        edges_new = torch.cat(edges, dim=1)
        num_nodes_cum = np.cumsum(num_nodes)

        body_ind = torch.from_numpy(np.concatenate(body_ind))
        if len(x) > 1:
            repeat_num = [x.shape[1] for x in edges[1:]]
            e_offset = np.repeat(num_nodes_cum[:-1], repeat_num)
            e_offset = torch.tensor(e_offset, device=obs.device)
            edges_new[:, -e_offset.shape[0]:] += e_offset


        return obs, edges_new, use_transform_action, num_nodes, num_nodes_cum, body_ind, hfield, obj, goal

    def process_hfield_obj(self, hfield, obj, goal, num_nodes):
        x_hfield = torch.stack(hfield, dim=0)
        x_hfield = self.hfield_enc(x_hfield)
        x_obj = torch.stack(obj, dim=0)
        x_obj = self.obj_enc(x_obj)
        x_goal = torch.stack(goal, dim=0)
        x_goal = self.goal_enc(x_goal)
        x_ext = torch.cat([x_hfield, x_obj, x_goal], dim=-1)
        x_ext = torch.repeat_interleave(x_ext, num_nodes, dim=0)
        return x_ext

    def forward(self, x):
        stages = ['skel_trans', 'attr_trans', 'execution']
        x_dict = defaultdict(list)
        node_design_mask = defaultdict(list)
        design_mask = defaultdict(list)
        total_num_nodes = 0
        for i, x_i in enumerate(x):
            num = x_i[3].item() # num_nodes
            cur_stage = stages[int(x_i[2].item())]
            x_dict[cur_stage].append(x_i)
            for stage in stages:
                node_design_mask[stage] += [cur_stage == stage] * num
                design_mask[stage].append(cur_stage == stage)
            total_num_nodes += num
        for stage in stages:
            node_design_mask[stage] = torch.BoolTensor(node_design_mask[stage])
            design_mask[stage] = torch.BoolTensor(design_mask[stage])

        # execution
        if len(x_dict['execution']) > 0:
            xe = x_dict['execution']
            obs, edges, _, num_nodes, num_nodes_cum_control, body_ind, hfield, obj, goal = self.batch_data(xe)
            self.body_ind_execution = body_ind
            x = self.control_norm(obs)
            if self.control_pre_mlp is not None:
                x = self.control_pre_mlp(x)
            if self.control_gnn is not None:
                x = self.control_gnn(x, edges)
            if self.control_mlp is not None:
                x = self.control_mlp(x)

            x_ext = self.process_hfield_obj(hfield, obj, goal, torch.from_numpy(num_nodes).to(x.device))
            x = torch.cat([x, x_ext], dim=-1)
            if self.control_ind_mlp is not None:
                control_action_mean = self.control_ind_mlp(x, body_ind)
            else:
                control_action_mean = self.control_action_mean(x)
            if self.shared_std_ctl == 1:
                control_action_std = self.control_action_log_std.expand_as(control_action_mean).exp()
            else:
                control_action_std = control_action_mean[:, self.control_action_dim:].exp()
                control_action_mean = control_action_mean[:, :self.control_action_dim]
            control_dist = DiagGaussian(control_action_mean, control_action_std)
        else:
            num_nodes_cum_control = None
            control_dist = None
            
        # attribute transform
        if len(x_dict['attr_trans']) > 0:
            obs, edges, _, num_nodes, num_nodes_cum_design, body_ind, hfield, obj, goal = self.batch_data(x_dict['attr_trans'])
            if self.enable_infer:
                embeds = self.group.get_cur_subgroup_embeds().unsqueeze(0).repeat(obs.shape[0], 1).to(obs.device)
                obs = torch.cat((obs[:, :self.attr_fixed_dim], obs[:, -self.attr_design_dim:], embeds), dim=-1)
            else:
                obs = torch.cat((obs[:, :self.attr_fixed_dim], obs[:, -self.attr_design_dim:]), dim=-1)
            # self.obs_attr_design = obs[:,-self.attr_design_dim:]
            self.body_ind_attribute = body_ind
            if self.attr_norm is not None:
                x = self.attr_norm(obs)
            else:
                x = obs
            if self.attr_pre_mlp is not None:
                x = self.attr_pre_mlp(x)
            if self.attr_gnn is not None:
                x = self.attr_gnn(x, edges)
            if self.attr_mlp is not None:
                x = self.attr_mlp(x)

            x_ext = self.process_hfield_obj(hfield, obj, goal, torch.from_numpy(num_nodes).to(x.device))
            x = torch.cat([x, x_ext], dim=-1)
            if self.attr_ind_mlp is not None:
                x = self.attr_ind_mlp(x, body_ind)
            else:
                x = self.attr_action_mean(x)

            attr_action_mean = torch.sin(x * (0.5 * np.pi))

            if self.enable_infer:
                attr_action_mean, _ = self.transform_policy_output(attr_action_mean, body_ind, attr=True)

            if self.shared_std_att == 1:
                attr_action_std = self.attr_action_log_std.expand_as(attr_action_mean).exp()
            else:
                attr_action_std = attr_action_mean[:, self.attr_action_dim:].exp()
                attr_action_mean = attr_action_mean[:, :self.attr_action_dim]
                # attr_action_std = attr_action_std / attr_action_std.mean().detach() * attr_action_mean.abs().mean().detach()
            attr_dist = DiagGaussian(attr_action_mean, attr_action_std)
        else:
            num_nodes_cum_design = None
            attr_dist = None

        # skeleleton transform
        if len(x_dict['skel_trans']) > 0:
            obs, edges, _, num_nodes, num_nodes_cum_skel, body_ind, hfield, obj, goal = self.batch_data(x_dict['skel_trans'])
            if self.enable_infer:
                embeds = self.group.get_cur_subgroup_embeds().unsqueeze(0).repeat(obs.shape[0],1).to(obs.device)
                obs = torch.cat((obs[:, :self.attr_fixed_dim], obs[:, -self.attr_design_dim:], embeds), dim=-1)
            else:
                obs = torch.cat((obs[:, :self.attr_fixed_dim], obs[:, -self.attr_design_dim:]), dim=-1)
            self.body_ind_skeleleton = body_ind
            x = self.skel_norm(obs)
            if self.skel_pre_mlp is not None:
                x = self.skel_pre_mlp(x)
            if self.skel_gnn is not None:
                x = self.skel_gnn(x, edges)

            if self.skel_mlp is not None:
                x = self.skel_mlp(x)
            x_ext = self.process_hfield_obj(hfield, obj, goal, torch.from_numpy(num_nodes).to(x.device))
            x = torch.cat([x,x_ext], dim=-1)
            if self.skel_ind_mlp is not None:
                skel_logits = self.skel_ind_mlp(x, body_ind)
            else:
                skel_logits = self.skel_action_logits(x)
            if self.enable_infer:
                skel_logits, _ = self.transform_policy_output(skel_logits, body_ind)

            skel_dist = Categorical(logits=skel_logits, uniform_prob=self.skel_uniform_prob)
        else:
            num_nodes_cum_skel = None
            skel_dist = None

        return control_dist, attr_dist, skel_dist, node_design_mask, design_mask, total_num_nodes, num_nodes_cum_control, num_nodes_cum_design, num_nodes_cum_skel, x[0][0].device

    def select_action(self, x, mean_action=False):
        
        control_dist, attr_dist, skel_dist, node_design_mask, _, total_num_nodes, _, _, _, device = self.forward(x)
        if control_dist is not None:
            control_action = control_dist.mean_sample() if mean_action else control_dist.sample()
            # control_action = control_action.clamp(-1, 1)
        else:
            control_action = None

        if attr_dist is not None:
            attr_action = attr_dist.mean_sample() if mean_action else attr_dist.sample()
            attr_action = torch.clamp(attr_action, -1, 1)
            if self.enable_infer:
                attr_action_env,  attr_action = self.transform_policy_output(attr_action, self.body_ind_attribute,
                                                                             attr=True, attr_select=True)
                self.is_new_morphology = False
            else:
                attr_action_env = attr_action
        else:
            attr_action = None

        if skel_dist is not None:
            skel_action = skel_dist.mean_sample() if mean_action else skel_dist.sample()
            skel_action[0] = 0
            skel_action_env = skel_action
            if self.enable_infer:
                skel_action_env, skel_action = self.transform_policy_output(skel_action, self.body_ind_skeleleton)
                self.is_new_morphology = True
        else:
            skel_action = None

        action = torch.zeros(total_num_nodes, self.action_dim).to(device)
        action_env = torch.zeros(total_num_nodes, self.action_dim).to(device)
        if control_action is not None:
            action[node_design_mask['execution'], :self.control_action_dim] = control_action
            action_env[node_design_mask['execution'], :self.control_action_dim] = control_action
        if attr_action is not None:
            action[node_design_mask['attr_trans'], self.control_action_dim:-1] = attr_action
            action_env[node_design_mask['attr_trans'], self.control_action_dim:-1] = attr_action_env
        if skel_action is not None:
            action[node_design_mask['skel_trans'], [-1]] = skel_action.double()
            action_env[node_design_mask['skel_trans'], [-1]] = skel_action_env.double()
        return action.numpy().astype(np.float64), action_env.numpy().astype(np.float64)

    def get_log_prob(self, x, action, get_entropy = False):
        action = torch.cat(action)
        control_dist, attr_dist, skel_dist, node_design_mask, design_mask, total_num_nodes, num_nodes_cum_control, num_nodes_cum_design, num_nodes_cum_skel, device = self.forward(x)
        action_log_prob = torch.zeros(design_mask['execution'].shape[0], 1).to(device)
        entropy = 0
        # execution log prob
        if control_dist is not None:
            control_action = action[node_design_mask['execution'], :self.control_action_dim]
            control_action_log_prob_nodes = control_dist.log_prob(control_action)
            control_action_log_prob_cum = torch.cumsum(control_action_log_prob_nodes, dim=0)
            control_action_log_prob_cum = control_action_log_prob_cum[torch.LongTensor(num_nodes_cum_control) - 1]
            control_action_log_prob = torch.cat([control_action_log_prob_cum[[0]], control_action_log_prob_cum[1:] - control_action_log_prob_cum[:-1]])
            action_log_prob[design_mask['execution']] = control_action_log_prob
            entropy += control_dist.entropy().mean()
        # attribute transform log prob
        if attr_dist is not None:
            attr_action = action[node_design_mask['attr_trans'], self.control_action_dim:-1]
            attr_action_log_prob_nodes = attr_dist.log_prob(attr_action)
            attr_action_log_prob_cum = torch.cumsum(attr_action_log_prob_nodes, dim=0)
            attr_action_log_prob_cum = attr_action_log_prob_cum[torch.LongTensor(num_nodes_cum_design) - 1]
            attr_action_log_prob = torch.cat([attr_action_log_prob_cum[[0]], attr_action_log_prob_cum[1:] - attr_action_log_prob_cum[:-1]])
            action_log_prob[design_mask['attr_trans']] = attr_action_log_prob
            entropy += attr_dist.entropy().mean()
        # skeleton transform log prob
        if skel_dist is not None:
            skel_action = action[node_design_mask['skel_trans'], [-1]]
            skel_action_log_prob_nodes = skel_dist.log_prob(skel_action)
            skel_action_log_prob_cum = torch.cumsum(skel_action_log_prob_nodes, dim=0)
            skel_action_log_prob_cum = skel_action_log_prob_cum[torch.LongTensor(num_nodes_cum_skel) - 1]
            skel_action_log_prob = torch.cat([skel_action_log_prob_cum[[0]], skel_action_log_prob_cum[1:] - skel_action_log_prob_cum[:-1]])
            action_log_prob[design_mask['skel_trans']] = skel_action_log_prob
            entropy += skel_dist.entropy().mean()

        if get_entropy:
            return action_log_prob, entropy
        else:
            return action_log_prob

    def transform_policy_output(self, outputs, body_ind, attr=False, attr_select=False): # only for ant
        base_repr = [np.base_repr(bi, self.index_base) for bi in body_ind]
        levels = np.array([len(bi) for bi in base_repr])
        level_0 = np.array([bi == '0' for bi in base_repr])
        levels[level_0] = 0
        body_ind_sym = torch.tensor([int(bi[-1]) for bi in base_repr], device=outputs.device)

        orbits = self.group.get_cur_orbits()
        for representative, orbit in orbits.items(): # only for scalars
            # body_ind_sym = body_ind % self.index_base
            orbit_select = (body_ind_sym == orbit[0])
            for i in orbit[1:]:
                orbit_select = torch.logical_or(orbit_select, (body_ind_sym == i))
            representative_select = (body_ind_sym == representative)
            orbit_size = len(orbit)

            for l in range(1, levels.max() + 1, 1):
                ind_l = levels == l
                ind_l_torch = torch.tensor(ind_l, device=outputs.device)
                representative_select_l = torch.logical_and(representative_select, ind_l_torch)
                repr_num_l = representative_select_l.sum().item()
                if repr_num_l == 0:
                    continue
                orbit_select_l = torch.logical_and(orbit_select, ind_l_torch)
                out_l = outputs[representative_select_l].reshape(repr_num_l, 1, -1).repeat(1, orbit_size, 1).reshape(outputs[orbit_select_l].shape)
                if attr:
                    if not self.shared_std_att == 1 and not attr_select:
                        if 'axis' in self.agent.cfg.robot_cfg['joint_params']:
                            slices = [ # (0,1,'2'),('3',4),5,6,
                                slice(2, 3),
                                slice(5, self.attr_action_dim),
                                slice(2 + self.attr_action_dim, 3 + self.attr_action_dim),
                                slice(5 + self.attr_action_dim, -1),
                            ]
                        else:
                            slices = [
                                slice(2, self.attr_action_dim),
                                slice(self.attr_action_dim + 2, -1),
                            ]
                    else:
                        if 'axis' in self.agent.cfg.robot_cfg['joint_params']:
                            slices = [ # (0,1,'2'),('3',4),5,6,
                                slice(2, 3),
                                slice(5, -1),
                            ]
                        else:
                            slices = [
                                slice(2, -1),
                            ]
                    for s in slices:
                        outputs[orbit_select_l, s] = out_l[:, s]
                else:
                    outputs[orbit_select_l] = out_l

        orbit_outputs = outputs.clone()
        if attr_select: # only for vectors
            symmetrizer = self.group.get_cur_subgroup_symmetrizer()

            for l in range(1, levels.max() + 1, 1):
                ind_l = levels == l
                body_l = body_ind_sym[ind_l] - 1

                if 'axis' in self.agent.cfg.robot_cfg['joint_params']: # (0,1,'2'),('3',4),5,6,
                    s = slice(0, 2)
                    out_l = sum([torch.tensor(e['matrix'], device=outputs.device) @ outputs[ind_l, s].t()
                                 @ e['permutation_inv'][body_l].reshape(len(body_l), -1)[:, body_l] * e['ratio'] for e
                                 in symmetrizer]).t()
                    outputs[ind_l, s] = out_l * np.sqrt(2) / 2


                else:
                    s = slice(0, 2)
                    out_l = sum([torch.tensor(e['matrix'], device=outputs.device) @ outputs[ind_l, s].t()
                                 @ e['permutation_inv'][body_l].reshape(len(body_l), -1)[:, body_l] * e['ratio'] for e
                                 in symmetrizer]).t()
                    outputs[ind_l, s] = out_l * np.sqrt(2) / 2

        return outputs, orbit_outputs

    def update_symmetry(self, episode_reward):
        self.group.update_subgroup(episode_reward)

    def get_info(self):
        return self.group.get_info()
