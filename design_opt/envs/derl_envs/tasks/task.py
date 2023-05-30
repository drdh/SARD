import design_opt.envs.derl_envs.utils.xml as xu
from design_opt.envs.derl_envs.config import cfg
from design_opt.envs.derl_envs.modules.agent import extract_agent_from_xml
from design_opt.envs.derl_envs.modules.agent import merge_agent_with_base
from design_opt.envs.derl_envs.tasks.escape_bowl import make_env_escape_bowl
from design_opt.envs.derl_envs.tasks.exploration import make_env_exploration
from design_opt.envs.derl_envs.tasks.incline import make_env_incline
from design_opt.envs.derl_envs.tasks.locomotion import make_env_locomotion
from design_opt.envs.derl_envs.tasks.manipulation import make_env_manipulation
from design_opt.envs.derl_envs.tasks.obstacle import make_env_obstacle
from design_opt.envs.derl_envs.tasks.patrol import make_env_patrol
from design_opt.envs.derl_envs.tasks.point_nav import make_env_point_nav
from design_opt.envs.derl_envs.tasks.push_box_incline import make_env_push_box_incline
from design_opt.envs.derl_envs.wrappers.select_keys import SelectKeysWrapper
from design_opt.envs.derl_envs.utils import file as fu
from design_opt.envs.derl_envs.wrappers.multi_env_wrapper import ToTransform2ActWrapper

def modify_xml_attributes(xml):
    root, tree = xu.etree_from_xml(xml, ispath=False)

    # Modify njmax and nconmax
    size = xu.find_elem(root, "size")[0]
    size.set("njmax", str(cfg.XML.NJMAX))
    size.set("nconmax", str(cfg.XML.NCONMAX))

    # Enable/disable filterparent
    flag = xu.find_elem(root, "flag")[0]
    flag.set("filterparent", str(cfg.XML.FILTER_PARENT))

    # Modify default geom params
    default_geom = xu.find_elem(root, "geom")[0]
    default_geom.set("condim", str(cfg.XML.GEOM_CONDIM))
    default_geom.set("friction", xu.arr2str(cfg.XML.GEOM_FRICTION))

    # Modify njmax and nconmax
    visual = xu.find_elem(root, "visual")[0]
    map_ = xu.find_elem(visual, "map")[0]
    map_.set("shadowclip", str(cfg.XML.SHADOWCLIP))

    return xu.etree_to_str(root)


def make_env(agent):
    ispath = False
    unimal_id = '0'

    xml = merge_agent_with_base(agent, ispath)
    xml = modify_xml_attributes(xml)

    env_func = "make_env_{}".format(cfg.ENV.TASK)
    env = globals()[env_func](xml, unimal_id)

    # Add common wrappers in the end
    keys_to_keep = cfg.ENV.KEYS_TO_KEEP + cfg.MODEL.OBS_TYPES
    env = SelectKeysWrapper(env, keys_to_keep=keys_to_keep)
    # env = ToTransform2ActWrapper(env)
    return env

