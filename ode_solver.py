
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torchdiffeq import odeint
from scipy.stats import qmc

from flow_matching.solver.solver import Solver
from flow_matching.utils import gradient, ModelWrapper
import numpy as np


# Generate RQMC Rademacher vectors
def generate_rqmc_vectors(shape, M, device, scramble, engine="sobol"):
    """Generate M quasi-random Rademacher vectors"""
    B, C, H, W = shape
    dimension = C * H * W  # We generate one vector per spatial location
    print(f"generating {engine} with d = {dimension}")

    # Create sequence generator
    if engine == "sobol":
        engine = qmc.Sobol(d=dimension, scramble=scramble)
    elif engine == "halton":
        engine = qmc.Halton(d=dimension, scramble=scramble)
    elif engine == "lhs":
        engine = qmc.LatinHypercube(d=dimension, scramble=scramble,
                                    optimization="random-cd")
    else:
        raise ValueError("Unknown engine given for RQMC vectors.")

    samples = engine.random(M)  # Shape: (M, dimension)

    # Convert to Rademacher: {-1, +1}
    rademacher_flat = 2 * (samples < 0.5).astype(np.float32) - 1

    # Reshape and convert to PyTorch tensors
    z_vectors = []
    for m in range(M):
        z = torch.from_numpy(rademacher_flat[m]).reshape(C, H, W).to(device)
        z_expanded = z.unsqueeze(0).expand(
            B, -1, -1, -1)  # Shape: (B, C, H, W)
        z_vectors.append(z_expanded)

    return z_vectors

# Taken (AND HEAVILY MODIFIED) from the flow matching library


class ODESolver(Solver):
    """A class to solve ordinary differential equations (ODEs) using a specified velocity model.

    This class utilizes a velocity field model to solve ODEs over a given time grid using numerical ode solvers.

    Args:
        velocity_model (Union[ModelWrapper, Callable]): a velocity field model receiving :math:`(x,t)` and returning :math:`u_t(x)`
    """

    def __init__(self, velocity_model: Union[ModelWrapper, Callable], args):
        super().__init__()
        self.velocity_model = velocity_model
        self.args = args
        self.z = None

    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tensor, Sequence[Tensor]]:
        r"""Solve the ODE with the velocity field.

        Example:

        .. code-block:: python

            import torch
            from flow_matching.utils import ModelWrapper
            from flow_matching.solver import ODESolver

            class DummyModel(ModelWrapper):
                def __init__(self):
                    super().__init__(None)

                def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                    return torch.ones_like(x) * 3.0 * t**2

            velocity_model = DummyModel()
            solver = ODESolver(velocity_model=velocity_model)
            x_init = torch.tensor([0.0, 0.0])
            step_size = 0.001
            time_grid = torch.tensor([0.0, 1.0])

            result = solver.sample(x_init=x_init, step_size=step_size, time_grid=time_grid)

        Args:
            x_init (Tensor): initial conditions (e.g., source samples :math:`X_0 \sim p`). Shape: [batch_size, ...].
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): The process is solved in the interval [min(time_grid, max(time_grid)] and if step_size is None then time discretization is set by the time grid. May specify a descending time_grid to solve in the reverse direction. Defaults to torch.tensor([0.0, 1.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Defaults to False.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Union[Tensor, Sequence[Tensor]]: The last timestep when return_intermediates=False, otherwise all values specified in time_grid.
        """

        time_grid = time_grid.to(x_init.device)

        def ode_func(t, x):
            return self.velocity_model(x=x, t=t, **model_extras)

        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            # Approximate ODE solution with numerical ODE solver
            sol = odeint(
                ode_func,
                x_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        if return_intermediates:
            return sol
        else:
            return sol[-1]

    def compute_likelihood(
        self,
        x_1: Tensor,
        log_p0: Callable[[Tensor], Tensor],
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([1.0, 0.0]),
        return_intermediates: bool = False,
        exact_divergence: bool = False,
        enable_grad: bool = False,
        calc_div=False,
        M=1,
        use_rqmc=False,
        **model_extras,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]:
        r"""Solve for log likelihood given a target sample at :math:`t=0`.

        Works similarly to sample, but solves the ODE in reverse to compute the log-likelihood. The velocity model must be differentiable with respect to x.
        The function assumes log_p0 is the log probability of the source distribution at :math:`t=0`.

        Args:
            x_1 (Tensor): target sample (e.g., samples :math:`X_1 \sim p_1`).
            log_p0 (Callable[[Tensor], Tensor]): Log probability function of the source distribution.
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): If step_size is None then time discretization is set by the time grid. Must start at 1.0 and end at 0.0, otherwise the likelihood computation is not valid. Defaults to torch.tensor([1.0, 0.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Otherwise only return the final sample. Defaults to False.
            exact_divergence (bool): Whether to compute the exact divergence or use the Hutchinson estimator.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            M (int, optional): Higher M leads to less variance for the trace prediction as O(1/M). But it also leads to VASTLY higher GPU memory requirements!
            use_rqmc (bool, optional): this toggles between vanilla hutchinson & randomized quasi monte carlo (RQMC)
            **model_extras: Additional input for the model.

        Returns:
            Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]: Samples at time_grid and log likelihood values of given x_1.
        """
        assert (
            time_grid[0] == 1.0 and time_grid[-1] == 0.0
        ), f"Time grid must start at 1.0 and end at 0.0. Got {time_grid}"

        # Fix the random projection for the Hutchinson divergence estimator
        if not exact_divergence and (self.z is None or self.args.batch_size != x_1[0].shape[0]):
            if M > 1:
                # NOTE: currently unused and not properly adapted to the rd_model
                if use_rqmc:

                    if self.args.rd_model:
                        self.z = [generate_rqmc_vectors(
                            x.shape, M, x.device, scramble=True, engine="lhs") for x in x_1]
                        # re-order to fit with other paradigm
                        self.z = [[self.z[idx_inner][idx_outer] for idx_inner in range(
                            len(x_1))] for idx_outer in range(M)]
                    else:
                        self.z = generate_rqmc_vectors(
                            x_1.shape, M, x_1.device, scramble=True)
                else:
                    if self.args.rd_model:
                        self.z = [[(torch.randn_like(x).to(x.device) < 0)
                                   * 2.0 - 1.0 for x in x_1] for _ in range(M)]
                    else:
                        self.z = [(torch.randn_like(x_1).to(x_1.device)
                                   < 0) * 2.0 - 1.0 for _ in range(M)]
            else:
                # if we use rd_model, z has to be prepared for several features
                if self.args.rd_model:
                    self.z = [(torch.randn_like(x).to(x.device) < 0)
                              * 2.0 - 1.0 for x in x_1]
                else:
                    self.z = (torch.randn_like(x_1).to(
                        x_1.device) < 0) * 2.0 - 1.0

        def ode_func(x, t):
            return self.velocity_model(x=x, t=t, **model_extras)

        flow_states = list()
        div_states = list()
        timesteps = list()

        def dynamics_func_tupled(t, states):
            # for rd4d-style models with three feature maps

            xt = states[:3]  # states[0]
            with torch.set_grad_enabled(True):
                for x in xt:
                    x.requires_grad_()
                ut = ode_func(xt, t)

                flow_states.append(xt)
                if not calc_div:
                    divs = [torch.zeros(
                        [u.shape[0], u.shape[-2], u.shape[-1]], device=u.device) for u in ut]
                    return tuple([u.detach() for u in ut] + divs)

                if M == 1:

                    ut_list, div_list = list(), list()
                    for x, u, cur_z in zip(xt, ut, self.z):
                        # Compute Hutchinson divergence estimator E[z^T D_x(ut) z]
                        # NOTE: Modififed for pixel-wise estimations! The spatial dimensions are kept intact
                        ut_dot_z = torch.einsum('ijkl,ijkl->ikl', u, cur_z)
                        # Compute the gradient of this dot product w.r.t. xt
                        grad_ut_dot_z = torch.autograd.grad(ut_dot_z, x, grad_outputs=torch.ones_like(ut_dot_z).detach(),
                                                            retain_graph=True, create_graph=True)[0]
                        # Compute the dot product of gradient and noise z, which estimates the divergence of the vector field ut
                        div = torch.einsum(
                            'ijkl,ijkl->ikl', grad_ut_dot_z, cur_z)

                        ut_list.append(u.detach())
                        div_list.append(div.detach())
                else:

                    ut_list, div_list = list(), list()
                    for hutch_idx, zippy in enumerate(zip(xt, ut)):
                        x, u = zippy
                        div_estimates = []
                        for m in range(M):
                            retain_graph = (m < M - 1) or hutch_idx < 2

                            cur_z_m = self.z[m][hutch_idx]
                            # Compute the dot product of ut and noise vector z (not flattened)
                            ut_dot_z = torch.einsum(
                                'ijkl,ijkl->ikl', u, cur_z_m)
                            # Compute the gradient of this dot product w.r.t. xt
                            grad_ut_dot_z = torch.autograd.grad(ut_dot_z, x, grad_outputs=torch.ones_like(ut_dot_z).detach(),
                                                                retain_graph=retain_graph, create_graph=True)[0]
                            # Compute the dot product of gradient and noise z, which estimates the divergence of the vector field ut
                            div = torch.einsum(
                                'ijkl,ijkl->ikl', grad_ut_dot_z, cur_z_m)

                            div_estimates.append(div)
                        div = torch.stack(div_estimates, dim=0).mean(dim=0)
                        ut_list.append(u.detach())
                        div_list.append(div.detach())

            div_states.append(div_list)

            return tuple(ut_list + div_list)

        def dynamics_func_single(t, states):
            # for regular single-feature u-net style networks
            xt = states[0]
            with torch.set_grad_enabled(True):
                xt.requires_grad_()
                ut = ode_func(xt, t)

                if not calc_div:
                    div = torch.zeros(
                        [ut.shape[0], ut.shape[-2], ut.shape[-1]], device=ut.device)
                    return ut.detach(), div

                if exact_divergence:
                    # Compute exact divergence
                    raise NotImplementedError(
                        "Currently, exact divergences are not implemented properly!")
                else:
                    # Compute Hutchinson divergence estimator E[z^T D_x(ut) z]
                    # NOTE: Modififed for pixel-wise estimations! The spatial dimensions are kept intact
                    if M == 1:
                        ut_dot_z = torch.einsum('ijkl,ijkl->ikl', ut, self.z)
                        # Compute the gradient of this dot product w.r.t. xt
                        grad_ut_dot_z = gradient(ut_dot_z, xt)
                        # Compute the dot product of gradient and noise z, which estimates the divergence of the vector field ut
                        div = torch.einsum(
                            'ijkl,ijkl->ikl', grad_ut_dot_z, self.z)
                    else:
                        # TODO: Could probably be vectorized
                        div_estimates = []
                        for m in range(M):
                            retain_graph = (m < M - 1)
                            cur_z = self.z[m]
                            # Compute the dot product of ut and noise vector z (not flattened)
                            ut_dot_z = torch.einsum(
                                'ijkl,ijkl->ikl', ut, cur_z)
                            # Compute the gradient of this dot product w.r.t. xt
                            grad_ut_dot_z = torch.autograd.grad(ut_dot_z, xt, grad_outputs=torch.ones_like(ut_dot_z).detach(),
                                                                retain_graph=retain_graph, create_graph=True)[0]
                            # Compute the dot product of gradient and noise z, which estimates the divergence of the vector field ut
                            div = torch.einsum(
                                'ijkl,ijkl->ikl', grad_ut_dot_z, cur_z)
                            div_estimates.append(div)
                        div = torch.stack(div_estimates, dim=0).mean(dim=0)

            return ut.detach(), div.detach()

        if self.args.rd_model:
            zeros = [torch.zeros(
                (x.shape[0], x.shape[2], x.shape[3]), device=x.device) for x in x_1]
            y_init = tuple(list(x_1) + zeros)
        else:
            y_init = (x_1, torch.zeros(
                (x_1.shape[0], x_1.shape[2], x_1.shape[3]), device=x_1.device))
        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            output = odeint(
                dynamics_func_single if not self.args.rd_model else dynamics_func_tupled,
                y_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        if self.args.rd_model:

            # NOTE: For the RD4AD-style model, we align likelihood scores from three different feature maps

            sol = output[:3]
            log_det = output[3:]
            # For WideResNet sol and log_det have outputs of shape [B, 64, 64], [B, 32, 32], [B, 16, 16]

            log_p0_terms = [log_p0(s[-1]) for s in sol]

            # NOTE: align all layers to first layer's dimensions
            interp_size = log_p0_terms[0].shape[1:]

            # NOTE: This is the for outputting the different maps
            log_p0_terms = [
                log_p0_terms[0][:, None],
                torch.nn.functional.interpolate(
                    log_p0_terms[1][:, None], interp_size, mode="bilinear"),
                torch.nn.functional.interpolate(
                    log_p0_terms[2][:, None], interp_size, mode="bilinear"),
            ]

            log_det_term = torch.concat([
                log_det[0][-1][:, None],
                torch.nn.functional.interpolate(
                    log_det[1][-1][:, None], interp_size, mode="bilinear"),
                torch.nn.functional.interpolate(
                    log_det[2][-1][:, None], interp_size, mode="bilinear"),
            ], 1)

            if calc_div:
                source_log_p = log_p0_terms[0].squeeze(
                    1) + log_det_term[:, 0, :, :]
                log_p0_term = log_p0_terms[1].squeeze(
                    1) + log_det_term[:, 1, :, :]
                log_det_term = log_p0_terms[2].squeeze(
                    1) + log_det_term[:, 2, :, :]
            else:
                # NOTE: we return anomaly maps of layers 0, 1, and 2
                source_log_p = log_p0_terms[0].squeeze(1)
                log_p0_term = log_p0_terms[1].squeeze(1)
                log_det_term = log_p0_terms[2].squeeze(1)

        else:

            # NOTE: in the simple single-scale U-Net case, it is simply the output of the ODE
            sol, log_det = output
            x_source = sol[-1]

            log_p0_term = log_p0(x_source)
            log_det_term = log_det[-1]
            source_log_p = log_p0_term + log_det_term

        if return_intermediates:
            return sol, source_log_p, log_p0_term, log_det_term
        else:
            return sol[-1], source_log_p, log_p0_term, log_det_term
