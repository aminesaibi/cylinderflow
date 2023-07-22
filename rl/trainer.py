import math
from tqdm import tqdm
import torch
from torch.optim import Adam
import torch.nn.functional as F
from utils import cast_dtype
from rl.memory import Memory
from copy import deepcopy


class Trainer():

    """Trainer Class"""

    def __init__(self,
                 model,
                 environment=None,
                 logger=None,
                 memory_size=20000,
                 params_update_rate=0.01,
                 policy_lr=1e-3,
                 value_lr=1e-2,
                 policy_reg=1e-5,
                 value_reg=1e-5,
                 batch_size=128,
                 n_episodes=5000,
                 n_rollout=200,
                 train_steps_per_episode=10,
                 id=0
                 ):
        """:attr """
        self.id = id
        self.device = model.device
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.batch_size = batch_size
        self.n_episodes = n_episodes
        self.n_rollout = n_rollout
        self.train_steps_per_episode = train_steps_per_episode
        self.params_update_rate = params_update_rate
        self.memory = Memory(memory_size)
        self.model = model
        self.environment = environment
        self.logger = logger
        self.target_model = deepcopy(model)
        self.policy_optimizer = Adam(model.fc_pi.parameters(),
                                     lr=policy_lr, weight_decay=policy_reg)
        self.value_optimizer = Adam(model.fc_q.parameters(),
                                    lr=value_lr, weight_decay=value_reg)
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.average_cost = 0.0
        self.best_cost = math.inf
        print('Trainer initialized.')

    def sample_batch(self):
        x1_batch = [None]*self.batch_size
        a_batch = [None]*self.batch_size
        r_batch = [None]*self.batch_size
        x2_batch = [None]*self.batch_size

        batch = self.memory.sample(self.batch_size)

        for i, transition in enumerate(batch):
            x1, a, r, x2 = transition
            x1_batch[i] = x1
            a_batch[i] = a
            r_batch[i] = r
            x2_batch[i] = x2

        x1_batch = torch.cat(x1_batch, dim=0)
        a_batch = torch.cat(a_batch, dim=0)
        r_batch = torch.cat(r_batch, dim=0)
        x2_batch = torch.cat(x2_batch, dim=0)

        return x1_batch, a_batch, r_batch, x2_batch

    def step(self):

        x1, a1, r, x2 = self.sample_batch()
        q_estim = self.model.q(x1, a1)
        a2 = self.target_model.pi(x2, deterministic=True)
        q_exp = r + self.model.gamma*self.target_model.q(x2, a2)

        value_loss = F.smooth_l1_loss(q_estim, q_exp.detach())
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        policy_loss = - self.target_model.q(x1,
                                            self.model.pi(x1, deterministic=True)
                                            ).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.soft_update(self.model, self.target_model, self.params_update_rate)

        return value_loss, policy_loss

    # def run_env(self, model=None, memory=None):
    #     if model is None:
    #         model = self.model
    #     if memory is None:
    #         model = self.memory

    #     # average_reward = 0.0
    #     # average_cost = 0.0
    #     # best_cost = math.inf
    #     for episode in tqdm(range(self.n_episodes), leave=False, position=self.id):
    #         with torch.no_grad():
    #             q0, q1 = self.environment.init_cond_generator()
    #             x1, _ = self.environment.sensors.get_values(q1)
    #             x1 = torch.tensor(x1).unsqueeze(0).to(self.device)

    #             for t in range(1, self.n_rollout+1):
    #                 a = model.pi(x1)
    #                 (q, _), x2, r, _ = self.environment.advance(u_n=q0,
    #                                                             u_n1=q1,
    #                                                             a=a.squeeze(0)
    #                                                             .detach()
    #                                                             .numpy()
    #                                                             )
    #                 x2 = torch.tensor(x2, device=self.device).unsqueeze(0)
    #                 r = torch.tensor(r).view(1, 1)

    #                 self.memory.append((x1, a, r, x2))
    #                 # average_reward = 1/t*((t-1)*average_reward + float(r))
    #                 # average_cost = 1/t*((t-1)*average_cost + float(E))
    #                 x1 = x2
    #                 q0 = q1
    #                 q1 = q

    #             # average_cost = 0.0
    #             # average_reward = 0.0
    #     # if self.logger is not None:
    #     #     if self.logger.proc is not None:
    #     #         self.logger.stopTensorBoard()

    def train(self, episode):
        if self.memory.size >= self.batch_size:
            progress = tqdm(desc="Training", total=self.train_steps_per_episode, leave=False, position=1)
            for _ in range(self.train_steps_per_episode):
                value_loss, policy_loss = self.step()
                self.logger.add([("Value Loss", value_loss),
                                 ("Policy Loss", policy_loss)
                                 # ("Average reward", average_reward),
                                 # ("Average cost", average_cost)
                                 ])
                progress.update(1)
            if self.logger is not None:
                self.logger.log(episode)
                torch.save(self.model.state_dict(),
                           self.logger.directory + "/model_state.pt")

                if self.average_cost < self.best_cost:
                    self.best_cost = self.average_cost
                    torch.save(self.model.state_dict(),
                               self.logger.directory + "/best_model.pt")
            # if verbose:
            #     print("Episode :{}, average_cost : {:.6f}, average_reward : {:.6f}, \
            #            value_loss : {:.6f}, \
            #            policy_loss : {:.6f}".format(episode,
            #                                         average_cost,
            #                                         average_reward,
            #                                         value_loss,
            #                                         policy_loss
            #                                         )
            #           )

    @staticmethod
    def soft_update(model, target_model, update_factor=0.1):
        """
        Copies the parameters from source network (x) to target network (y) update
        y = update_factor*x + (1 - update_factor)*y
        :param target: Target network
        :param source: Source network
        :return:
        """
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - update_factor) +
                                    param.data * update_factor
                                    )
