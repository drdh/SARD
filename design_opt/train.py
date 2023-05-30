import argparse
import os
import sys
sys.path.append(os.getcwd())

from khrylib.utils import *
from design_opt.utils.config import Config
from design_opt.agents.transform2act_agent import Transform2ActAgent
import datetime
import wandb
from design_opt.envs.derl_envs.config import cfg as derl_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None)
parser.add_argument("--derl_cfg", dest="cfg_file", help="Config file", required=False, type=str)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--tmp', action='store_true', default=False)
parser.add_argument('--num_threads', type=int, default=20)
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--epoch', default='0')
parser.add_argument('--show_noise', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--enable_wandb', type=int, default=0)
parser.add_argument('--enable_infer', type=int, default=0)
parser.add_argument('--expID', type=str, default=None)
args = parser.parse_args()

enable_wandb = bool(args.enable_wandb)
if enable_wandb:
    wandb.init(project="design-rl", entity="drdh")
enable_infer = bool(args.enable_infer)

if args.expID is None:
    running_start_time = datetime.datetime.now()
    args.expID = str(running_start_time.strftime("%Y_%m_%d-%X"))

if args.render:
    args.num_threads = 1
if args.cfg_file is not None:
    derl_cfg.merge_from_file(args.cfg_file)
# cfg = Config(args.cfg, args.tmp, create_dirs=not (args.render or args.epoch != '0'))
cfg = Config(args.cfg, args.tmp, expID=args.expID, enable_infer=enable_infer, derl_task=derl_cfg.ENV.TASK)


cfg.seed = args.seed
cfg.enable_wandb = enable_wandb
if enable_wandb:
    wandb.run.name = args.expID
    wandb.config.update(cfg)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

start_epoch = int(args.epoch) if args.epoch.isnumeric() else args.epoch

"""create agent"""
agent = Transform2ActAgent(cfg=cfg, dtype=dtype, device=device, seed=cfg.seed, num_threads=args.num_threads, training=True, checkpoint=start_epoch)


def main_loop():

    if args.render:
        agent.pre_epoch_update(start_epoch)
        agent.sample(1e8, mean_action=not args.show_noise, render=True)
    else:
        for epoch in range(start_epoch, cfg.max_epoch_num):          
            agent.optimize(epoch)
            agent.save_checkpoint(epoch)

            """clean up gpu memory"""
            torch.cuda.empty_cache()

        agent.logger.info('training done!')


main_loop()
if enable_wandb:
    wandb.finish()