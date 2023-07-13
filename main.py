import argparse
import torch
from environment.ns import NS
from utils import *
from rl.model import Model
import multiprocessing as mp
torch.set_default_dtype(torch.float64)


class Main():
    def __init__(self, queue=None, device="cpu"):
        self.queue = queue
        self.device = device

    def run(self, args):
        if args.mode[0] == "train":
            from rl.trainer import Trainer
            env_params = load_yaml(args.env[0])
            environment = NS(**env_params)

            model_params = load_yaml(args.model_params)
            model = Model(environment.ns,
                          environment.na,
                          **model_params,
                          device=self.device
                          )

            trainer_params = load_yaml(args.mode[1])
            configs = {"env_params": env_params,
                       "model_params": model_params,
                       "trainer_params": trainer_params}

            logger = Logger("./logs", configs)
            trainer = Trainer(model, environment, logger, **trainer_params)
            logger.loadTensorBoard()
            trainer.run()
        else:
            raise Exception(
                "Please specify a valid mode : train.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--test", action='store_true')
    parser.add_argument("-e", "--env", nargs=1, type=str)
    parser.add_argument("-p", "--model_params", type=str)
    parser.add_argument("-m", "--mode", nargs="+", type=str)
    parser.add_argument("-d", "--device", nargs="?",
                        const="cpu", default="cpu", type=str)
    args = parser.parse_args()
    if (args.model_params is None) or (args.env is None) or (args.mode is None):
        parser.error("execution requires --mode, --model_params and --env.")
    else:
        manager = mp.Manager()
        queue = manager.Queue()
        main = Main(queue, device=args.device)
        main.run(args)
