import concurrent.futures
import io
import dill
import argparse
import subprocess
from tqdm import tqdm
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor, MPICommExecutor
from ns import NS
from utils import *
from rl.model import Model
import multiprocessing as mp
from petsc4py import PETSc
import pickle
torch.set_default_dtype(torch.float64)

# class Main():
#     def __init__(self,  device="cpu"):
#         self.device = device
#         # Initialize PETSc and MPI
#         # PETSc.Sys.popErrorHandler()
#         # PETSc.initialize()
#         # self.comm = PETSc.COMM_WORLD.tompi4py()
#         self.comm = MPI.COMM_WORLD
#         self.petsc_comm = self.comm.Dup()
#         self.size = self.comm.Get_size()
#         self.rank = self.comm.Get_rank()
#         env_params = load_yaml(args.env[0])
#         self.environment = NS(**env_params, comm=self.petsc_comm)
#         model_params = load_yaml(args.model_params)
#         self.model = Model(self.environment.ns,
#                            self.environment.na,
#                            **model_params,
#                            device=self.device
#                            )
#         if self.rank == 0:
#             from rl.trainer import Trainer
#             trainer_params = load_yaml(args.mode[1])
#             configs = {"env_params": env_params,
#                        "model_params": model_params,
#                        "trainer_params": trainer_params}
#             logger = Logger("./logs", configs)
#             self.trainer = Trainer(self.model, self.environment, logger=logger, **trainer_params)
#             logger.loadTensorBoard()

#     def generate_transitions(self, model, n_rollout):
#         return self.environment.run_model((model, n_rollout, self.rank))
#         # transitions = comm.gather(transitions, root=0)
#         # self.comm.send(transitions, dest=0)

#     def run(self, args):
#         if args.mode[0] == "train":
#             progress = tqdm(desc="Episode", total=100, leave=True, position=0)
#             for episode in range(100):
#                 transitions = self.generate_transitions(self.model, 200)
#                 if self.rank == 0:
#                     self.trainer.memory.append_list(transitions)
#                     self.trainer.train(episode)
#                     progress.update(1)
#                 self.model = self.comm.bcast(self.model, root=0)
#                 self.comm.Barrier()
#             # PETSc.finalize()
#             # MPI.Finalize()
#         elif args.mode[0] == "test":
#             torch.manual_seed(747)
#             np.random.seed(747)
#             env_params = load_yaml(args.env[0])
#             environment = NS(**env_params, mpi_comm=comm)

#             model_params = load_yaml(args.model_params)
#             model = Model(environment.ns,
#                           environment.na,
#                           **model_params,
#                           device=self.device
#                           )
#             model.load_state_dict(torch.load(args.mode[1],
#                                   map_location=model.device))
#             tmax = int(args.mode[2])
#             environment.run(T=tmax, controller=model.pi)
# def run(self, args):
#     if args.mode[0] == "train":
#         from rl.trainer import Trainer
#         env_params = load_yaml(args.env[0])
#         environment = NS(**env_params, comm=self.comm)

#         model_params = load_yaml(args.model_params)
#         model = Model(environment.ns,
#                       environment.na,
#                       **model_params,
#                       device=self.device
#                       )

#         trainer_params = load_yaml(args.mode[1])
#         # configs = {"env_params": env_params,
#         #            "model_params": model_params,
#         #            "trainer_params": trainer_params}

#         # logger = Logger("./logs", configs)
#         if self.comm.rank == 0:
#             trainer = Trainer(model, environment, logger=None, **trainer_params)
#             # logger.loadTensorBoard()
#             trainer.run_env()
#             trainer.train()
#     else:
#         raise Exception(
#             "Please specify a valid mode : train.")
#######################################################################################
#     if args.mode[0] == "train":
#         env_params = load_yaml(args.env[0])
#         environment = NS(**env_params, comm=self.comm)
#         if self.comm.rank == 0:
#             from rl.trainer import Trainer
#             model_params = load_yaml(args.model_params)
#             model = Model(environment.ns,
#                           environment.na,
#                           **model_params,
#                           device=self.device
#                           )

#             trainer_params = load_yaml(args.mode[1])
#             configs = {"env_params": env_params,
#                        "model_params": model_params,
#                        "trainer_params": trainer_params}

#             logger = Logger("./logs", configs)
#             trainer = Trainer(model, environment, logger=logger, **trainer_params)
#             memory = trainer.memory
#             logger.loadTensorBoard()
#         for episode in tqdm(self.trainer.n_episodes):
#             self.environment.run_model(model, memory, trainer.n_rollout, model.device)
#             if self.comm.rank == 0:
#                 trainer.train(episode)
#     else:
#         raise Exception(
#             "Please specify a valid mode : train.")
#############################################################


class Main():
    def __init__(self,  device="cpu"):
        self.device = device

    # def generate_transitions(self, model, n_rollout):
    #     return self.environment.run_model((model, n_rollout, self.rank))
    #     # transitions = comm.gather(transitions, root=0)
    #     # self.comm.send(transitions, dest=0)
    def run_env(self, args):
        tmp_file, n_rollout = args
        child_process_command = ['mpirun', '-np', '9', "python", "ns.py",
                                 tmp_file,
                                 str(n_rollout)
                                 ]
        completed_process = subprocess.check_output(child_process_command, text=False)
        return completed_process

    def run(self, args):
        if args.mode[0] == "train":
            tmp_file = './tmp/'
            env_params = load_yaml(args.env[0])
            write_yaml(os.path.join(tmp_file, 'env_params'), env_params)

            model_params = load_yaml(args.model_params)
            model = Model(len(env_params['xs']),
                          len(env_params['xa']),
                          **model_params,
                          device=self.device
                          )
            write_yaml(os.path.join(tmp_file, 'model_params'), model_params)
            torch.save(model.state_dict(), tmp_file+"/state_dict.pt")

            from rl.trainer import Trainer
            trainer_params = load_yaml(args.mode[1])
            configs = {"env_params": env_params,
                       "model_params": model_params,
                       "trainer_params": trainer_params}
            logger = Logger("./logs", configs)
            trainer = Trainer(model, logger=logger, **trainer_params)
            logger.loadTensorBoard()
            progress = tqdm(desc="Episode", total=100, leave=True, position=0)
            for episode in range(100):
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    future = executor.submit(self.run_env, (tmp_file, trainer.n_rollout))
                completed_process = future.result()
                byte_transition = (completed_process.decode('latin1').split("'\n")[-1]).encode('latin1')
                transitions_from_child = pickle.loads(byte_transition)

                trainer.memory.append_list(transitions_from_child)
                trainer.train(episode)
                torch.save(model.state_dict(), tmp_file+"/state_dict.pt")
                progress.update(1)

        elif args.mode[0] == "test":
            torch.manual_seed(747)
            np.random.seed(747)
            env_params = load_yaml(args.env[0])
            environment = NS(**env_params, mpi_comm=comm)

            model_params = load_yaml(args.model_params)
            model = Model(environment.ns,
                          environment.na,
                          **model_params,
                          device=self.device
                          )
            model.load_state_dict(torch.load(args.mode[1],
                                  map_location=model.device))
            tmax = int(args.mode[2])
            environment.run(T=tmax, controller=model.pi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--test", action='store_true')
    parser.add_argument("-e", "--env", nargs=1, type=str)
    parser.add_argument("-p", "--model_params", type=str)
    parser.add_argument("-m", "--mode", nargs="+", type=str)
    parser.add_argument("-d", "--device", nargs="?",
                        const="cpu", default="cpu", type=str)
    args = parser.parse_args()
    if ((args.model_params is None) or (args.env is None) or (args.mode is None)) and False:
        parser.error("execution requires --mode, --model_params and --env.")
    else:
        main = Main(device=args.device)
        main.run(args)
        #############################################################################################
        # device = 'cpu'
        # if args.mode[0] == "train":
        #     env_params = load_yaml(args.env[0])
        #     environment = NS(**env_params)
        #     model_params = load_yaml(args.model_params)
        #     model = Model(environment.ns,
        #                   environment.na,
        #                   **model_params,
        #                   device=device
        #                   )

        #     trainer_params = load_yaml(args.mode[1])
        #     configs = {"env_params": env_params,
        #                "model_params": model_params,
        #                "trainer_params": trainer_params}

        #     logger = Logger("./logs", configs)
        #     from rl.trainer import Trainer
        #     trainer = Trainer(model, environment, logger=logger, **trainer_params)
        #     logger.loadTensorBoard()
        #     for episode in tqdm(range(100)):
        #         trainer.memory = environment.run_model(trainer.model, trainer.memory, 500)
        #         trainer.train(episode)
        # elif args.mode[0] == "test":
        #     torch.manual_seed(747)
        #     np.random.seed(747)
        #     env_params = load_yaml(args.env[0])
        #     environment = NS(**env_params)

        #     model_params = load_yaml(args.model_params)
        #     model = Model(environment.ns,
        #                   environment.na,
        #                   **model_params,
        #                   device=device
        #                   )
        #     model.load_state_dict(torch.load(args.mode[1],
        #                           map_location=model.device))
        #     tmax = int(args.mode[2])
        #     environment.run(T=tmax, controller = model.pi)
        ################################################################
        # device = 'cpu'
        # if args.mode[0] == "train":
        #     env_params = load_yaml(args.env[0])
        #     environment = NS(**env_params)
        #     model_params = load_yaml(args.model_params)
        #     model = Model(environment.ns,
        #                   environment.na,
        #                   **model_params,
        #                   device=device
        #                   )

        #     trainer_params = load_yaml(args.mode[1])
        #     configs = {"env_params": env_params,
        #                "model_params": model_params,
        #                "trainer_params": trainer_params}

        #     logger = Logger("./logs", configs)
        #     from rl.trainer import Trainer
        #     trainer = Trainer(model, environment, logger=logger, **trainer_params)
        #     logger.loadTensorBoard()
        #     for episode in tqdm(range(100)):
        #         transitions = [None]*trainer.n_rollout
        #         # comm = MPI.COMM_WORLD
        #         # with MPICommExecutor(comm, root=0) as executor:
        #         executor = MPIPoolExecutor(max_workers=10)
        #         future = executor.map(environment.run_model,
        #                               (trainer.model, 500))

        #         # assert future.done()
        #         # transitions = future.result()
        #         # comm.send(transitions, dest=0, tag=47)
        #         # for rk in range(1, comm.Get_size()):
        #         # transitions = comm.recv(transitions, source=rk, tag=47)
        #         for transitions in future:
        #             print(transitions[0])
        #             # trainer.memory.append_list(transitions)
        #             # trainer.train(episode)
        #         # comm.bcast(trainer, root=0)
        # elif args.mode[0] == "test":
        #     torch.manual_seed(747)
        #     np.random.seed(747)
        #     env_params = load_yaml(args.env[0])
        #     environment = NS(**env_params)

        #     model_params = load_yaml(args.model_params)
        #     model = Model(environment.ns,
        #                   environment.na,
        #                   **model_params,
        #                   device=device
        #                   )
        #     model.load_state_dict(torch.load(args.mode[1],
        #                           map_location=model.device))
        #     tmax = int(args.mode[2])
        #     environment.run(T=tmax, controller=model.pi)
