import torch
from torch import nn, Tensor

from discit.accel import capture_graph
from discit.distr import MultiNormal, FixedVarNormal
from discit.func import symexp
from discit.optim import NAdamW, AdaptivePlateauScheduler
from discit.rl import ActorCritic, PPG
from discit.track import CheckpointTracker


class CartpoleEnv:
    """
    Reference:
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    """

    N_STEPS_PER_SECOND = 32
    DT = 1. / N_STEPS_PER_SECOND
    MAX_DURATION = 16.

    RESET_DIST = 2.4
    RESET_ANGLE = torch.pi / 4.

    G = 9.8
    CART_MASS = 1.
    POLE_MASS = 0.1
    TOTAL_MASS = CART_MASS + POLE_MASS

    POLE_HALF_LEN = 0.5
    POLE_MASS_MUL_HALF_LEN = POLE_MASS * POLE_HALF_LEN

    FORCE_SCALE = 10.

    _2PI = 2. * torch.pi
    _4_DIV_3 = 4. / 3.
    NULL_DICT = {}

    def __init__(self, n_envs: int = 1, device: str = 'cuda'):
        self.n_envs = n_envs
        self.device = device

        self.pos, self.vel, self.angle, self.ang_vel, self.was_upright, self.duration = self.random_init(n_envs)

    def random_init(self, n_envs: int) -> 'tuple[Tensor, ...]':
        pos, vel, angle, ang_vel = torch.empty((4, n_envs), device=self.device).uniform_(-0.05, 0.05)

        angle *= torch.pi / 0.05

        was_upright = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
        duration = torch.zeros(n_envs, device=self.device)

        return pos, vel, angle, ang_vel, was_upright, duration

    def reset(self, rst_idcs: Tensor = None, obs: 'tuple[Tensor, Tensor]' = None):
        if rst_idcs is None:
            rst_idcs = torch.arange(self.n_envs, device=self.device)

        pos, vel, angle, ang_vel, was_upright, duration = self.random_init(len(rst_idcs))

        self.pos[rst_idcs] = pos
        self.vel[rst_idcs] = vel
        self.angle[rst_idcs] = angle
        self.ang_vel[rst_idcs] = ang_vel
        self.was_upright[rst_idcs] = was_upright
        self.duration[rst_idcs] = duration

        if obs is not None:
            obs[0][rst_idcs] = torch.stack((
                pos,
                vel,
                angle.cos(),
                angle.sin(),
                ang_vel), dim=1)

            obs[1][rst_idcs] = torch.stack((
                was_upright.float(),
                1. - duration / self.MAX_DURATION), dim=1)

    def check_reset(self) -> Tensor:
        rst_pos = self.pos.abs() > self.RESET_DIST
        rst_ang = self.was_upright & (self.angle.abs() > self.RESET_ANGLE)
        rst_dur = self.duration > self.MAX_DURATION

        return (rst_pos | rst_ang | rst_dur).float()

    def get_observation(self) -> Tensor:
        obs_vec = torch.stack((
            self.pos,
            self.vel,
            self.angle.cos(),
            self.angle.sin(),
            self.ang_vel), dim=1)

        obs_aux = torch.stack((
            self.was_upright.float(),
            1. - self.duration / self.MAX_DURATION), dim=1)

        return obs_vec, obs_aux

    def get_reward(self) -> Tensor:
        rew_vel = self.vel.square().neg_().exp_()       # Max. at 0. vel.
        rew_ang_vel = self.pos.square().neg_().exp_()   # Max. at 0. ang. vel.
        rew_ang = self.angle.cos().add_(1.).div_(2.)    # Max. at 0. angle

        return rew_vel.mul_(rew_ang_vel).mul_(rew_ang)

    def step(self, action: Tensor = None) -> 'tuple[tuple[Tensor, ...], Tensor, Tensor, dict]':
        if action is None:
            return self.get_observation(), None, None, self.NULL_DICT

        *obs, rew, rst = self.partial_step(action.flatten())
        rst_idcs = torch.nonzero(rst, as_tuple=True)[0]

        if len(rst_idcs):
            self.reset(rst_idcs, obs)

        return obs, rew, rst, self.NULL_DICT

    def partial_step(self, action: Tensor) -> 'tuple[Tensor, ...]':
        force = action * self.FORCE_SCALE
        angle_cos = self.angle.cos()
        angle_sin = self.angle.sin()

        tmp = (force + self.POLE_MASS_MUL_HALF_LEN * self.ang_vel**2 * angle_sin) / self.TOTAL_MASS

        ang_acc = (
            (self.G * angle_sin - angle_cos * tmp)
            / (self.POLE_HALF_LEN * (self._4_DIV_3 - self.POLE_MASS * angle_cos**2 / self.TOTAL_MASS)))

        acc = tmp - self.POLE_MASS_MUL_HALF_LEN * ang_acc * angle_cos / self.TOTAL_MASS

        self.vel += self.DT * acc
        self.pos += self.DT * self.vel
        self.ang_vel += self.DT * ang_acc
        self.duration += self.DT

        new_angle = self.angle + self.DT * self.ang_vel

        corr_mask_gt = new_angle > torch.pi
        corr_mask_lt = new_angle < -torch.pi
        non_corr_mask = ~(corr_mask_gt | corr_mask_lt)

        new_angle += (corr_mask_lt.float() - corr_mask_gt.float()) * self._2PI

        self.was_upright |= (new_angle.sign() != self.angle.sign()) & non_corr_mask
        self.angle.copy_(new_angle)

        obs = self.get_observation()
        rew = self.get_reward()
        rst = self.check_reset()

        return *obs, rew, rst


# Overkill for cartpole, but the point is to test the components, not just solve cartpole
class CartpoleModel(ActorCritic):
    def __init__(
        self,
        n_in_vec: int = 5,      # pos., vel., ang. cos., ang. sin., ang. vel.
        n_in_aux: int = 2,      # was_upright, duration
        enc_size: int = 64,
        mem_size: int = 64,
        n_actions: int = 1,     # Horizontal force
        n_values: int = 1,
        chrono_len: int = max(2, CartpoleEnv.N_STEPS_PER_SECOND // 8)   # 1/8th of a second
    ):
        super().__init__()

        self.activ = nn.Tanh()

        self.fcin = nn.Linear(n_in_vec, enc_size)
        self.rnnp = nn.GRUCell(enc_size, mem_size)
        self.fcout = nn.Linear(mem_size, 2*n_actions)

        self.rnnv = nn.GRUCell(mem_size + n_in_aux, mem_size)
        self.fcv = nn.Linear(mem_size, n_values)

        self.memp = nn.Parameter(torch.zeros(1, mem_size).uniform_(-1., 1.))
        self.memv = nn.Parameter(torch.zeros(1, mem_size).uniform_(-1., 1.))

        nn.init.orthogonal_(self.fcin.weight)
        nn.init.zeros_(self.fcin.bias)
        nn.init.orthogonal_(self.rnnp.weight_ih)
        nn.init.orthogonal_(self.rnnp.weight_hh)
        nn.init.zeros_(self.rnnp.bias_ih)
        nn.init.zeros_(self.rnnp.bias_hh)
        nn.init.orthogonal_(self.fcout.weight, gain=0.001)
        nn.init.zeros_(self.fcout.bias)

        nn.init.orthogonal_(self.rnnv.weight_ih)
        nn.init.orthogonal_(self.rnnv.weight_hh)
        nn.init.zeros_(self.rnnv.bias_ih)
        nn.init.zeros_(self.rnnv.bias_hh)
        nn.init.orthogonal_(self.fcv.weight, gain=0.001)
        nn.init.zeros_(self.fcv.bias)

        with torch.no_grad():
            self.rnnp.bias_hh[mem_size:-mem_size].uniform_(1, chrono_len - 1).log_()
            self.rnnv.bias_hh[mem_size:-mem_size].uniform_(1, chrono_len - 1).log_()

    def init_mem(self, batch_size: int = 1, detach: bool = False) -> 'tuple[Tensor]':
        memp = self.memp.expand(batch_size, -1)
        memv = self.memv.expand(batch_size, -1)

        if detach:
            memp = memp.detach().clone()
            memv = memv.detach().clone()

        return memp, memv

    def reset_mem(
        self,
        mem: 'tuple[Tensor]',
        reset_mask: Tensor
    ) -> 'tuple[Tensor]':

        reset_mask = reset_mask[..., None]
        memp, memv = mem

        memp = torch.lerp(memp, self.memp, reset_mask)
        memv = torch.lerp(memv, self.memv, reset_mask)

        return memp, memv

    def get_distr(self, args: 'tuple[Tensor, ...]') -> MultiNormal:
        return MultiNormal(*args)

    def fwd_partial(
        self,
        obs_vec: Tensor,
        obs_aux: Tensor,
        memp: Tensor,
        memv: Tensor
    ) -> 'tuple[Tensor, ...]':

        with torch.no_grad():
            x = self.activ(self.fcin(obs_vec))
            memp = self.rnnp(x, memp)
            x = self.fcout(memp)

            v = torch.cat((memp, obs_aux), dim=1)
            memv = self.rnnv(v, memv)
            v = self.fcv(memv)

            val_mean = FixedVarNormal(symexp(v)).mean.flatten()

        return x, val_mean, memp, memv

    def fwd_partial_copied(
        self,
        obs_vec: Tensor,
        obs_aux: Tensor,
        memp: Tensor,
        memv: Tensor
    ) -> 'tuple[Tensor, ...]':

        return self.fwd_partial(obs_vec, obs_aux, memp, memv)

    def fwd_actor(*args):
        raise NotImplementedError

    def fwd_critic(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]'
    ) -> 'tuple[Tensor, tuple[Tensor, ...]]':

        _, val_mean, memp, memv = self.fwd_partial(*obs, *mem)

        return val_mean, (memp, memv)

    def fwd_collector(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]'
    ) -> 'tuple[tuple[Tensor, ...], tuple[Tensor, ...], Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]':

        x, val_mean, memp, memv = self.fwd_partial(*obs, *mem)

        act = MultiNormal.from_raw(x[:, :1], x[:, 1:])

        return act.sample(), (act.mean, act.log_dev), val_mean, (obs[0].clone(), obs[1].clone()), (memp, memv)

    def fwd_recollector(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        get_distr: bool = True
    ) -> 'tuple[tuple[Tensor, ...], Tensor, tuple[Tensor, ...]]':

        x, val_mean, memp, memv = self.fwd_partial_copied(*obs, *mem)

        if get_distr:
            act = MultiNormal.from_raw(x[:, :1], x[:, 1:])
            act_args = (act.mean, act.log_dev)

        else:
            act_args = ()

        return act_args, val_mean, (memp, memv)

    def fwd_learner(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]'
    ) -> 'tuple[MultiNormal, FixedVarNormal, tuple[Tensor, ...]]':

        x = self.activ(self.fcin(obs[0]))
        memp = self.rnnp(x, mem[0])
        x = self.fcout(memp)

        v = torch.cat((memp, obs[1]), dim=1)
        memv = self.rnnv(v, mem[1])
        v = self.fcv(memv)

        act = MultiNormal.from_raw(x[:, :1], x[:, 1:])
        val = FixedVarNormal(symexp(v))

        return act, val, (memp, memv)


if __name__ == '__main__':
    n_envs = 256

    epoch_milestones = [2, 30, 32]
    n_rollout_steps = 256
    n_truncated_steps = 16
    n_main_iters = 8
    n_aux_iters = 6

    n_updates_per_epoch = (
        n_rollout_steps // n_truncated_steps * n_main_iters
        + n_rollout_steps * n_main_iters // n_truncated_steps * n_aux_iters)

    # Init envs.
    ckpter = CheckpointTracker('cartpolew')
    env = CartpoleEnv(n_envs, ckpter.device)

    # Init model
    model = CartpoleModel()
    optimiser = NAdamW(model.parameters(), lr=2e-5, weight_decay=0., clip_grad_value=None)

    scheduler = AdaptivePlateauScheduler(
        optimiser,
        step_milestones=[ep * n_updates_per_epoch for ep in epoch_milestones],
        starting_step=ckpter.meta['update_step'])

    # Load last weights
    model.to(ckpter.device)
    ckpter.load_model(model, optimiser)

    # Accelerate env. step
    accelerate = True

    if accelerate:
        inputs = torch.zeros(n_envs, dtype=torch.float32, device=ckpter.device),

        env.partial_step, env_step_graph = capture_graph(env.partial_step, inputs, copy_idcs_out=(2, 3))
        env.reset()

        # Accelerate collector, recollector, and critic
        mem = model.init_mem(n_envs, detach=True)
        inputs = (*env_step_graph['out'][:2], *mem)

        fwd_partial_copied = model.fwd_partial
        model.fwd_partial, fwd_partial_graph = capture_graph(model.fwd_partial, inputs, copy_idcs_in=(2, 3))

        mem = model.init_mem(n_envs, detach=True)
        inputs = (*[torch.rand_like(o) for o in env_step_graph['out'][:2]], *mem)

        model.fwd_partial_copied, fwd_partial_copied_graph = capture_graph(fwd_partial_copied, inputs)

    rl_algo = PPG(
        env.step,
        ckpter,
        scheduler,
        n_epochs=epoch_milestones[-1],
        log_epoch_interval=1,
        ckpt_epoch_interval=0,
        branch_epoch_interval=0,
        n_rollout_steps=n_rollout_steps,
        n_truncated_steps=n_truncated_steps,
        batch_size=n_envs,
        n_main_iters=n_main_iters,
        n_aux_iters=n_aux_iters,
        discount_factor=0.98,
        entropy_weight=0.,
        accelerate=accelerate,
        update_returns=False)

    try:
        rl_algo.run()
        print('\nDone.')

    except KeyboardInterrupt:
        print('\nProcess aborted due to user command.')

    rl_algo.writer.close()
