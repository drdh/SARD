import numpy as np
import os
import os.path as osp
import gym
from gym import utils, error, spaces
from gym.utils import seeding
import mujoco_py
from khrylib.rl.envs.common.mujoco_env_gym import MujocoEnv, convert_observation_to_space
from khrylib.robot.xml_robot import Robot
from khrylib.utils import get_single_body_qposaddr, get_graph_fc_edges
from khrylib.utils.transformation import quaternion_matrix
import shutil

from design_opt.envs.derl_envs.tasks.task import modify_xml_attributes
from design_opt.envs.derl_envs.modules.agent import extract_agent_from_xml
from design_opt.envs.derl_envs.modules.agent import merge_agent_with_base
from design_opt.envs.derl_envs.config import cfg as derl_cfg

from design_opt.envs.derl_envs.tasks.escape_bowl import make_env_escape_bowl
from design_opt.envs.derl_envs.tasks.exploration import make_env_exploration
from design_opt.envs.derl_envs.tasks.incline import make_env_incline
from design_opt.envs.derl_envs.tasks.locomotion import make_env_locomotion
from design_opt.envs.derl_envs.tasks.manipulation import make_env_manipulation
from design_opt.envs.derl_envs.tasks.obstacle import make_env_obstacle
from design_opt.envs.derl_envs.tasks.patrol import make_env_patrol
from design_opt.envs.derl_envs.tasks.point_nav import make_env_point_nav
from design_opt.envs.derl_envs.tasks.push_box_incline import make_env_push_box_incline
from design_opt.envs.derl_envs.morphology import Morphology
from design_opt.envs.derl_envs.utils import mjpy as mu

# DEFAULT_SIZE = 500
DEFAULT_SIZE = 1024
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

class DerlEnv(gym.Env, utils.EzPickle):
    def __init__(self, cfg, agent):
        self.cur_t = 0
        self.cfg = cfg
        self.env_specs = cfg.env_specs
        self.agent = agent
        self.enable_design = self.cfg.enable_design
        min_n_legs = min(cfg.num_legs) # 3
        self.max_n_legs = cur_n_legs = max_n_legs = max(cfg.num_legs) #6

        self.model_xml_file = cfg.init_design_xml_file

        derl_cfg.ENV.WALKER = self.unimal_ant_xml_file = "design_opt/envs/derl_envs/assets/unimal_ant.xml"
        derl_cfg.freeze()

        # robot xml
        self.init_xml_str_list = dict()
        xml_file = f'assets/mujoco_envs/ant{max_n_legs}.xml'
        cur_robot = Robot(cfg.robot_cfg, xml=xml_file)
        for n_leg in range(max_n_legs, min_n_legs - 1, -1):
            while cur_n_legs != n_leg:
                if cur_n_legs < n_leg:
                    raise NotImplementedError
                elif cur_n_legs > n_leg:
                    cur_robot.remove_body(cur_robot.bodies[1])
                    cur_n_legs -= 1
            self.init_xml_str_list[n_leg] = cur_robot.export_xml_string()
        self.robot = Robot(cfg.robot_cfg, xml=self.model_xml_file)
        self.init_xml_str = self.robot.export_xml_string()
        self.cur_xml_str = self.init_xml_str.decode('utf-8')
        # design options
        self.clip_qvel = cfg.obs_specs.get('clip_qvel', False)
        self.use_projected_params = cfg.obs_specs.get('use_projected_params', True)
        self.abs_design = cfg.obs_specs.get('abs_design', True)
        self.use_body_ind = cfg.obs_specs.get('use_body_ind', False)
        self.task = 'direction'
        self._reset_task()
        self.design_ref_params = self.get_attr_design()
        self.design_cur_params = self.design_ref_params.copy()
        self.design_param_names = self.robot.get_params(get_name=True)
        self.attr_design_dim = self.design_ref_params.shape[-1]
        self.index_base = max(5, max_n_legs + 1)
        self.stage = 'skeleton_transform'  # transform or execute
        self.control_nsteps = 0
        self.sim_specs = set(cfg.obs_specs.get('sim', []))
        self.attr_specs = set(cfg.obs_specs.get('attr', []))
        # Simmulator
        self.frame_skip = 4
        self.reset_cur_derl_env()
        self.model = self.derl_env.model
        self.sim = self.derl_env.sim
        self.data = self.derl_env.sim.data

        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.is_inited = False

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)
        self.seed()
        self.is_inited = True

        utils.EzPickle.__init__(self)
        self.control_action_dim = 1
        self.skel_num_action = 3 if cfg.enable_remove else 2
        self.sim_obs_dim = self.get_sim_obs().shape[-1]
        self.sim_obs_hfield_dim = self.get_sim_obs_hfield().shape[0]
        self.sim_obs_obj_dim = self.get_sim_obs_obj().shape[0]
        self.sim_obs_goal_dim = self.get_sim_obs_goal().shape[0]

        self.attr_fixed_dim = self.get_attr_fixed().shape[-1]
        
    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space
    
    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reload_sim_model(self, xml_str):
        del self.sim
        del self.model
        del self.data
        del self.viewer
        del self._viewers
        self.reset_cur_derl_env()
        self.model = self.derl_env.model
        self.sim = self.derl_env.sim
        self.data = self.derl_env.sim.data
        self.init_qpos = self.sim.data.qpos.copy()
        self.init_qvel = self.sim.data.qvel.copy()
        self.viewer = None
        self._viewers = {}

    def reset_cur_derl_env(self):
        xml = Morphology('0', self.robot).to_string()
        self.derl_env = gym.make('Unimal-v0', agent=xml)
        self.sim_obs = self.derl_env.reset()


    def allow_add_body(self, body):
        add_body_condition = self.cfg.add_body_condition
        max_nchild = add_body_condition.get('max_nchild', 3)
        min_nchild = add_body_condition.get('min_nchild', 0)
        return body.depth >= self.cfg.min_body_depth and body.depth < self.cfg.max_body_depth - 1 and len(
            body.child) < max_nchild and len(body.child) >= min_nchild

    def allow_remove_body(self, body):
        if body.depth >= self.cfg.min_body_depth + 1 and len(body.child) == 0:
            if body.depth == 1:
                return body.parent.child.index(body) > 0
            else:
                return True
        return False

    def apply_skel_action(self, skel_action):
        if self.enable_design:
            bodies = list(self.robot.bodies)
            for body, a in zip(bodies, skel_action):
                if a == 1 and self.allow_add_body(body):
                    self.robot.add_child_to_body(body)
                if a == 2 and self.allow_remove_body(body):
                    self.robot.remove_body(body)

        xml_str = self.robot.export_xml_string()
        self.cur_xml_str = xml_str.decode('utf-8')
        self.design_cur_params = self.get_attr_design()
        return True

    def set_design_params(self, in_design_params):
        if self.enable_design:
            design_params = in_design_params
            for params, body in zip(design_params, self.robot.bodies):
                body.set_params(params, pad_zeros=True, map_params=False)
                body.sync_node()

        xml_str = self.robot.export_xml_string()
        self.cur_xml_str = xml_str.decode('utf-8')
        try:
            self.reload_sim_model(xml_str.decode('utf-8'))
        except:
            print(self.cur_xml_str)
            return False
        if self.use_projected_params:
            self.design_cur_params = self.get_attr_design()
        else:
            self.design_cur_params = in_design_params.copy()
        return True

    def action_to_control(self, a):
        ctrl = a.squeeze()[self.o2d][1:]
        return ctrl

    def step(self, a):
        if not self.is_inited:
            return self._get_obs(zeros=False), 0, False, {'use_transform_action': False, 'stage': 'execution'}

        self.cur_t += 1
        # skeleton transform stage
        if self.stage == 'skeleton_transform':
            skel_a = a[:, -1]
            succ = self.apply_skel_action(skel_a)
            if not succ:
                return self._get_obs(zeros=True), 0.0, True, {'use_transform_action': True, 'stage': 'skeleton_transform'}

            if self.cur_t == self.cfg.skel_transform_nsteps:
                self.transit_attribute_transform()

            ob = self._get_obs(zeros=True)
            reward = 0.0
            done = False
            return ob, reward, done, {'use_transform_action': True, 'stage': 'skeleton_transform'}
        # attribute transform stage
        elif self.stage == 'attribute_transform':
            design_a = a[:, self.control_action_dim:-1]
            if self.abs_design:
                design_params = design_a * self.cfg.robot_param_scale
            else:
                design_params = self.design_cur_params + design_a * self.cfg.robot_param_scale
            succ = self.set_design_params(design_params)
            if not succ:
                return self._get_obs(zeros=False), 0.0, True, {'use_transform_action': True, 'stage': 'attribute_transform'}

            if self.cur_t == self.cfg.skel_transform_nsteps + 1:
                succ = self.transit_execution()
                if not succ:
                    return self._get_obs(zeros=False), 0.0, True, {'use_transform_action': True, 'stage': 'attribute_transform'}

            ob = self._get_obs(zeros=False)
            reward = 0.0
            done = False
            return ob, reward, done, {'use_transform_action': True, 'stage': 'attribute_transform'}
        # execution stage
        else:
            self.control_nsteps += 1
            assert np.all(a[:, self.control_action_dim:] == 0)
            control_a = a[:, :self.control_action_dim]
            ctrl = self.action_to_control(control_a)
            self.sim_obs, reward, done, info = self.derl_env.step(ctrl)

            if not done:
                reward += 0.01

            ob = self._get_obs(zeros=False)
            return ob, reward, done, {'use_transform_action': False, 'stage': 'execution'}

    def transit_attribute_transform(self):
        self.stage = 'attribute_transform'

    def transit_execution(self):
        self.stage = 'execution'
        self.control_nsteps = 0
        try:
            self.reset_state(True)
        except:
            print(self.cur_xml_str)
            return False
        return True

    def if_use_transform_action(self):
        return ['skeleton_transform', 'attribute_transform', 'execution'].index(self.stage)

    def get_sim_obs(self, zeros = False):
        if zeros:
            obs = np.zeros((len(self.robot.bodies), self.sim_obs_dim))
        else:
            obs = self.sim_obs['proprioceptive']
            select_obs = ~self.sim_obs['obs_padding_mask']
            obs = obs.reshape(len(select_obs),-1)[select_obs]
            derl_bodies = mu.names_from_prefixes(self.sim, ["torso", "limb"], "body")
            o2d = {0:0,}
            d2o = {0:0,}
            for o in list(range(1,len(derl_bodies))):
                d = derl_bodies.index(f"limb/{o-1}")
                o2d[o] = d
                d2o[d] = o
            self.o2d = np.array([o2d[o] for o in range(len(o2d))])
            self.d2o = np.array([d2o[d] for d in range(len(d2o))])
            obs = obs[self.d2o]
        return obs

    def get_sim_obs_hfield(self, zeros = False):
        if 'hfield' not in self.sim_obs:
            obs = np.zeros(1)
        elif zeros:
            obs = np.zeros(self.sim_obs_hfield_dim)
        else:
            obs = self.sim_obs['hfield']
        return obs

    def get_sim_obs_obj(self, zeros = False):
        if 'obj' not in self.sim_obs:
            obs = np.zeros(1)
        elif zeros:
            obs = np.zeros(self.sim_obs_obj_dim)
        else:
            obs = self.sim_obs['obj']
        return obs

    def get_sim_obs_goal(self, zeros = False):
        if 'goal' not in self.sim_obs:
            obs = np.zeros(1)
        elif zeros:
            obs = np.zeros(self.sim_obs_goal_dim)
        else:
            obs = self.sim_obs['goal']
        return obs



    def get_attr_fixed(self):
        obs = []
        for i, body in enumerate(self.robot.bodies):
            obs_i = []
            if 'depth' in self.attr_specs:
                obs_depth = np.zeros(self.cfg.max_body_depth)
                obs_depth[body.depth] = 1.0
                obs_i.append(obs_depth)
            if 'jrange' in self.attr_specs:
                obs_jrange = body.get_joint_range()
                obs_i.append(obs_jrange)
            if 'skel' in self.attr_specs:
                obs_add = self.allow_add_body(body)
                obs_rm = self.allow_remove_body(body)
                obs_i.append(np.array([float(obs_add), float(obs_rm)]))

            for j in range(self.cfg.max_body_depth):
                obs_id = np.zeros(self.index_base)
                if len(body.name) > j:
                    obs_id[int(body.name[-(j + 1)], base=36)] = 1.0
                obs_i.append(obs_id)

            if len(obs_i) > 0:
                obs_i = np.concatenate(obs_i)
                obs.append(obs_i)

        if len(obs) == 0:
            return None
        obs = np.stack(obs)
        return obs

    def get_attr_design(self):
        obs = []
        for i, body in enumerate(self.robot.bodies):
            obs_i = body.get_params([], pad_zeros=True, demap_params=True)
            obs.append(obs_i)
        obs = np.stack(obs)
        return obs

    def get_body_index(self):
        index = []
        for i, body in enumerate(self.robot.bodies):
            ind = int(body.name, base=self.index_base)
            index.append(ind)
        index = np.array(index)
        return index

    def _get_obs(self, zeros=False):
        attr_fixed_obs = self.get_attr_fixed()  # only depth
        sim_obs = self.get_sim_obs(zeros=zeros)  # [qpos, qvel]
        design_obs = self.design_cur_params
        # Do not change the order or add new items in the concatenate.
        obs = np.concatenate(list(filter(lambda x: x is not None, [attr_fixed_obs, sim_obs, design_obs])), axis=-1)
        if self.cfg.obs_specs.get('fc_graph', False):
            edges = get_graph_fc_edges(len(self.robot.bodies))
        else:
            edges = self.robot.get_gnn_edges()
        use_transform_action = np.array([self.if_use_transform_action()])
        num_nodes = np.array([sim_obs.shape[0]])
        all_obs = [obs, edges, use_transform_action, num_nodes]  # Do not change it.
        if self.use_body_ind:
            body_index = self.get_body_index()
            all_obs.append(body_index)
        all_obs.append(self.get_sim_obs_hfield(zeros=zeros))
        all_obs.append(self.get_sim_obs_obj(zeros=zeros))
        all_obs.append(self.get_sim_obs_goal(zeros=zeros))
        return all_obs

    def _reset_task(self):
        if self.task == 'forward':
            pass
        elif self.task == 'direction':
            self.target_direction_ang = np.random.uniform(0, 2 * np.pi)
            self.target_direction_vec = np.array([np.cos(self.target_direction_ang), np.sin(self.target_direction_ang)])
            self.extra_shared_obs = np.array([np.cos(self.target_direction_ang), np.sin(self.target_direction_ang)])
        elif self.task == 'LRward':
            self.target_direction_ang = np.random.choice([0.0, np.pi])
            self.target_direction_vec = np.array([np.cos(self.target_direction_ang), np.sin(self.target_direction_ang)])
            self.extra_shared_obs = np.array([np.cos(self.target_direction_ang), np.sin(self.target_direction_ang)])
        else:
            raise NotImplementedError

    def reset_state(self, add_noise):
        pass

    def reset_robot(self, extra_args=None):
        del self.robot
        if extra_args is not None and self.enable_design:
            self.init_xml_str = self.init_xml_str_list[extra_args['num']]
        self.robot = Robot(self.cfg.robot_cfg, xml=self.init_xml_str, is_xml_str=True)
        self.cur_xml_str = self.init_xml_str.decode('utf-8')
        # self.reload_sim_model(self.cur_xml_str)
        self.design_ref_params = self.get_attr_design()
        self.design_cur_params = self.design_ref_params.copy()

    def reset_model(self, extra_args=None):
        self._reset_task()
        self.reset_robot(extra_args=extra_args)
        self.control_nsteps = 0
        self.stage = 'skeleton_transform'
        self.cur_t = 0
        return self._get_obs(zeros=True)
    
    def reset(self, extra_args = None):
        ob = self.reset_model(extra_args = extra_args)
        return ob
    
    def set_state(self, qpos, qvel):
        self.derl_env.set_state(qpos, qvel)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip
    
    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()
            
    def render(self,
               mode='human',
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=None,
               camera_name=None):
        self.derl_env.render(mode=mode, width=width, height=height, camera_id=camera_id, camera_name=camera_name)

    def close(self):
        self.derl_env.close()


    def _get_viewer(self, mode):
        return self.derl_env.get_viewer(mode)

    def viewer_setup(self):
        self.derl_env.viewer_setup()


    def set_custom_key_callback(self, key_func):
        self._get_viewer('human').custom_key_callback = key_func

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def vec_body2world(self, body_name, vec):
        body_xmat = self.data.get_body_xmat(body_name)
        vec_world = (body_xmat @ vec[:, None]).ravel()
        return vec_world

    def pos_body2world(self, body_name, pos):
        body_xpos = self.data.get_body_xpos(body_name)
        body_xmat = self.data.get_body_xmat(body_name)
        pos_world = (body_xmat @ pos[:, None]).ravel() + body_xpos
        return pos_world






