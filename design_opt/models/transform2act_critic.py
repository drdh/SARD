import torch.nn as nn
import torch
import numpy as np
from khrylib.models.mlp import MLP
from khrylib.rl.core.running_norm import RunningNorm
from design_opt.utils.tools import *
from design_opt.models.gnn import GNNSimple
from design_opt.models.jsmlp import ObsEncoder



class Transform2ActValue(nn.Module):
    def __init__(self, cfg, agent):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.enable_infer = agent.cfg.enable_infer
        self.design_flag_in_state = cfg.get('design_flag_in_state', False)
        self.onehot_design_flag = cfg.get('onehot_design_flag', False)
        self.sym_embed_dim = sum(agent.cfg.num_legs) * 2
        if self.enable_infer:
            self.state_dim = agent.state_dim + self.design_flag_in_state * (3 if self.onehot_design_flag else 1) + self.sym_embed_dim
        else:
            self.state_dim = agent.state_dim + self.design_flag_in_state * (3 if self.onehot_design_flag else 1)
        self.sim_obs_hfield_dim = agent.sim_obs_hfield_dim
        self.sim_obs_obj_dim = agent.sim_obs_obj_dim
        self.sim_obs_goal_dim = agent.sim_obs_goal_dim
        self.norm = RunningNorm(self.state_dim)
        cur_dim = self.state_dim
        self.hfield_enc = ObsEncoder(self.sim_obs_hfield_dim)
        self.obj_enc = ObsEncoder(self.sim_obs_obj_dim)
        self.goal_enc = ObsEncoder(self.sim_obs_goal_dim)
        shared_obs_dim = self.hfield_enc.out_dim + self.obj_enc.out_dim + self.goal_enc.out_dim
        if 'pre_mlp' in cfg:
            self.pre_mlp = MLP(cur_dim, cfg['pre_mlp'], cfg['htype'])
            cur_dim = self.pre_mlp.out_dim
        else:
            self.pre_mlp = None
        if 'gnn_specs' in cfg:
            self.gnn = GNNSimple(cur_dim, cfg['gnn_specs'])
            cur_dim = self.gnn.out_dim
        else:
            self.gnn = None
        if 'mlp' in cfg:
            self.mlp = MLP(cur_dim, cfg['mlp'], cfg['htype'])
            cur_dim = self.mlp.out_dim
        else:
            self.mlp = None
        cur_dim += shared_obs_dim
        self.value_head = nn.Linear(cur_dim, 1)
        init_fc_weights(self.value_head)


        if self.enable_infer:
            self.group = agent.policy_net.group


    def batch_data(self, x):
        if len(x[0]) == 8:
            obs, edges, use_transform_action, num_nodes, body_ind, hfield, obj, goal = zip(*x)
        else:
            raise NotImplementedError
            # obs, edges, use_transform_action, num_nodes = zip(*x)
        obs = torch.cat(obs)
        use_transform_action = np.concatenate(use_transform_action)
        num_nodes = np.concatenate(num_nodes)
        edges_new = torch.cat(edges, dim=1)
        if len(x) > 1:
            repeat_num = [x.shape[1] for x in edges[1:]]
            num_nodes_cum = np.cumsum(num_nodes)
            e_offset = np.repeat(num_nodes_cum[:-1], repeat_num)
            e_offset = torch.tensor(e_offset, device=obs.device)
            edges_new[:, -e_offset.shape[0]:] += e_offset
        else:
            num_nodes_cum = None
        return obs, edges_new, use_transform_action, num_nodes, num_nodes_cum, hfield, obj, goal

    def forward(self, x):
        obs, edges, use_transform_action, num_nodes, num_nodes_cum, hfield, obj, goal = self.batch_data(x)
        if self.enable_infer:
            embeds = self.group.get_cur_subgroup_embeds().unsqueeze(0).repeat(obs.shape[0], 1).to(obs.device)
            obs = torch.cat((obs, embeds), dim=-1)
        if self.design_flag_in_state:
            design_flag = torch.tensor(np.repeat(use_transform_action, num_nodes)).to(obs.device)
            if self.onehot_design_flag:
                design_flag_onehot = torch.zeros(design_flag.shape[0], 3).to(obs.device)
                design_flag_onehot.scatter_(1, design_flag.unsqueeze(1), 1)
                x = torch.cat([obs, design_flag_onehot], dim=-1)
            else:
                x = torch.cat([obs, design_flag.unsqueeze(1)], dim=-1)
        else:
            x = obs
        x = self.norm(x)
        if self.pre_mlp is not None:
            x = self.pre_mlp(x)
        if self.gnn is not None:
            x = self.gnn(x, edges)
        if self.mlp is not None:
            x = self.mlp(x)

        x_ext = self.process_hfield_obj(hfield, obj, goal, torch.from_numpy(num_nodes).to(x.device))
        x = torch.cat([x, x_ext], dim=-1)
        x = self.value_head(x)
        if num_nodes_cum is None:
            x = x[[0]]
        else:
            x = x[torch.LongTensor(np.concatenate([np.zeros(1), num_nodes_cum[:-1]]))]
        return x

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