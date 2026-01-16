import numpy as np
import copy
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import solve_ivp
import pickle
from scipy.interpolate import interp1d
import datetime
import warnings

# warnings.filterwarnings("ignore", message="xxx")
TIME_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

class MergedSolution:
    '''Merge multiple odesolver solutions (e.g. daily solutions in a daily dilution scenario) into one solution.'''

    def __init__(self, solutions):
        assert all(hasattr(sol, 't') and hasattr(sol, 'y') for sol in
                   solutions), "All inputs must be valid solution instances."

        # Sort solutions by the first element in their t (time) array
        sorted_solutions = sorted(solutions, key=lambda sol: sol.t[0])

        self.t_events = []
        self.status = []
        self.message = []
        self.success = []
        self.nfev = 0
        self.njev = 0
        self.nlu = 0

        for sol in sorted_solutions:
            if hasattr(sol, 't_events'): self.t_events.append(sol.t_events)
            if hasattr(sol, 'status'): self.status.append(sol.status)
            if hasattr(sol, 'message'): self.message.append(sol.message)
            if hasattr(sol, 'success'): self.success.append(sol.success)
            if hasattr(sol, 'nfev'): self.nfev += sol.nfev
            if hasattr(sol, 'njev'): self.njev += sol.njev
            if hasattr(sol, 'nlu'): self.nlu += sol.nlu

        # Merge t and y
        t, y = [], []
        for sol in sorted_solutions:
            for t_val, y_val in zip(sol.t, sol.y.T):
                t.append(t_val)
                y.append(y_val)
        self.t = np.array(t)
        self.y = np.array(y).T


class ConsumerResourceModel:
    """ A class to define a consumer-resource model.

    Model Parameters:
    * nS: number of species
    * nR: number of resources
    * v0: maximum uptake rate matrix (nS x nR)
    * km0: half-saturation constant matrix (nS x nR) under baseline conditions
    * r0: maximum growth rate vector (nS)
    * e: conversion efficiency matrix (nS x nR)
    * b: cross-feeding matrix (nS x nR x nR), where b_ijk specifies fraction of unusable resource j converted to resource k by species i
    * gamma: resource diversity penalty parameter (nS), controls how km changes with resource diversity

    Key Model Features:

    Resource Uptake:
    * Species i uptakes resource j with Monod kinetics: v_ij = v0_ij * R_j / (km_ij + R_j)
    * Biomass conversion efficiency e_ij
    * Half-saturation constant km can be modified by:
        - Default: km_ij = km0_ij / (1 + gamma_i * (nR_consumed - 1))
        - Custom: provide fun_km(self, R, Rsupp) function
    * Conversion efficiency e can be modified by:
        - Default: use fixed e matrix
        - Custom: provide fun_e(self, R, Rsupp) function

    Growth Dynamics (two modes):
    * 'mod': Growth rate capped by r0
        - Realized growth rate: a_i = sum_j(e_ij * v_ij)
        - Actual growth rate: r_i = min(a_i, r0_i)
        - Adjusted consumption: c_ij = v_ij * r_i / a_i
        - Species dynamics: dN_i/dt = r_i * N_i - dil * N_i
    * 'mod2' (default): Unlimited growth rate
        - Species dynamics: dN_i/dt = sum_j(e_ij * v_ij) * N_i - dil * N_i

    Resource Dynamics:
    * dR_j/dt = -uptake_j + secretion_j - dil * R_j + dil * Rsupp_j
    * Uptake: uptake_j = sum_i(c_ij * N_i) for 'mod' or sum_i(v_ij * N_i) for 'mod2'
    * Secretion: secretion_k = sum_ij(b_ijk * (1 - e_ij) * consumption_ij * N_i)

    Cross-feeding:
    * If e_ij < 1, unused fraction (1 - e_ij) is secreted as other resources
    * Distribution controlled by b_ijk matrix, where sum(b_ijk, axis=2) = 1

    Dilution:
    * Continuous: constant dilution rate dil throughout simulation
    * Daily: discrete dilution events at interval T (default 24h)
        - Relationship: dil_daily = 1 - (1 - dil_continuous)^(1/T)

    Additional Methods:
    * sim(): simulate with continuous dilution
    * sim_daily_dil(): simulate with daily discrete dilution
    * __add__() or concat_models(): combine multiple models (concatenate species)
    * subset(): extract submodel with selected species and/or resources
    * get_parameters(): return model parameters as dictionary

    """

    def __init__(self, nS, nR, v0, km0, r0, e, b, gamma=0):
        """Initialize the model
        """
        self.nS = nS
        self.nR = nR

        # check if the input parameters are valid
        assert v0.shape == (nS, nR), f"v0 must be a {nS} x {nR} matrix, currently {v0.shape}"
        assert km0.shape == (nS, nR), f"km0 must be a {nS} x {nR} matrix, currently {km0.shape}"
        assert len(r0) == nS, f"r0 must be a {nS} vector, currently {r0.shape}"
        r0 = np.array(r0).reshape(-1, 1)
        assert e.shape == (nS, nR), f"e must be a {nS} x {nR} matrix, currently {e.shape}"
        assert b.shape == (nS, nR, nR), f"b must be a {nS} x {nR} x {nR} matrix, currently {b.shape}"
        if np.isscalar(gamma):  # if gamma is a scalar, broadcast to a vector
            gamma = np.repeat(gamma, nS)
        assert len(gamma) == nS, f"gamma must be a {nS} vector, currently {gamma.shape}"
        gamma = np.array(gamma).reshape(-1, 1)

        self.v0 = v0
        self.km0 = km0
        self.r0 = r0
        self.e = e
        self.b = b
        self.gamma = gamma

    # return parameters as a dictionary
    def get_parameters(self):
        """Return the model parameters as a dictionary
        """
        params = {
            'nS': self.nS,
            'nR': self.nR,
            'v0': self.v0,
            'km0': self.km0,
            'r0': self.r0,
            'e': self.e,
            'b': self.b,
            'gamma': self.gamma
        }
        return params

    # Define the differential equations
    def model(self, t, y, dil, Rsupp, nRsupp=None, fun_km=None, fun_e=None, use_model='mod2'):
        """Define the ode-model for the consumer-resource system

        """
        N, R = copy.deepcopy(y[:self.nS]), copy.deepcopy(y[self.nS:])
        # treat negative values (occur due to numerical accuracy limit) as 0
        N[N < 0] = 0
        N = N.reshape([-1, 1])  # nS * 1
        R[R < 0] = 0
        R = R.reshape([1, -1])  # 1 * nR

        if nRsupp is None:
            nRsupp = np.sum(Rsupp > 0)

        # ----- Defining how gamma impact species under different resource supply -----
        if fun_km is None:
            nRconsum_curr = np.sum(self.v0 * Rsupp > 0.0001, axis=1, keepdims=True)  # number of R consumed by each species in given Rsupp
            nRconsum_curr = np.maximum(nRconsum_curr, 1)  # at least 1 to avoid division by 0. when nRconsum_curr==0 the species can't grow anyway
            km = self.km0 / (1 + self.gamma * (nRconsum_curr - 1))  # this way km0 is the km as defined in single resource condition
        else:
            km = fun_km(self, R, Rsupp)

        if fun_e is None:
            e = self.e
        else:
            e = fun_e(self, R, Rsupp)
        # ----- End of definition -----

        km = np.maximum(km, 1e-6)  # avoid numerical error
        v = self.v0 * R / (km + R)  # nS * nR

        if use_model == 'mod':  # growth rate capped by r0
            a = np.sum(e * v, axis=1, keepdims=True)  # nS * 1, realized growth rate. maximum equals number of resource
            r = np.minimum(a, self.r0)
            dNdt = r * N - dil * N
            c = v * r / (a + 1e-10)  # nS * nR, consumption
            uptake = np.sum(c * N, axis=0, keepdims=True)  # nR * 1
            secretion = np.sum(np.sum(self.b * ((1 - e) * c * N)[:, :, None], axis=0), axis=0, keepdims=True) # nR * 1. sum over spe then res
            dRdt = -uptake + secretion - dil * R + dil * Rsupp
        elif use_model == 'mod2':  # no growth rate cap
            dNdt = np.sum(e * v, axis=1, keepdims=True) * N - dil * N
            uptake = np.sum(v * N, axis=0, keepdims=True)  # nR * 1
            secretion = np.sum(np.sum(self.b * ((1 - e) * v * N)[:, :, None], axis=0), axis=0,
                               keepdims=True)  # nR * 1. sum over spe then res
            dRdt = -uptake + secretion - dil * R + dil * Rsupp
        else:
            raise ValueError(f"use_model {use_model} not recognized. Must be 'mod' or 'mod2'.")
        # # print shapes
        # print("uptake shape: ", uptake.shape)
        # print("secretion shape: ", secretion.shape)
        # print("dRdt shape: ", dRdt.shape)
        # print("R shape: ", R.shape)
        # print("N shape: ", N.shape)
        # print("dNdt shape: ", dNdt.shape)
        # print("Rsupp shape: ", Rsupp.shape)

        return np.concatenate([dNdt.flatten(), dRdt.flatten()])


    # Simulate the model
    def sim(self, t, N0, R0, dil, Rsupp, nRsupp=None, **kwargs):
        """
        Simulate the consumer-resource model with continuous dilution

        Parameters:
        * t: time points to evaluate the solution.
            t can be [tini, tend] or a list of time points to evaluate the solution
        * N0: initial species abundances (nS,)
        * R0: initial resource concentrations (nR,)
        * dil: continuous dilution rate
        * Rsupp: resource supply concentrations (nR,)
        * nRsupp: number of resources being supplied. If None, inferred from Rsupp
        * kwargs: additional arguments for scipy.integrate.solve_ivp

        """
        # if kwargs doesn't specify atol and rtol, use default values
        if 'atol' not in kwargs:
            kwargs['atol'] = 1e-6
        if 'rtol' not in kwargs:
            kwargs['rtol'] = 1e-3
        # if kwargs doesn't specify method, use RK45
        if 'method' not in kwargs:
            kwargs['method'] = 'RK45'
        # if kwargs doesn't specify the ODE model to use, use 'mod0' - model()
        if 'model' not in kwargs:
            use_model = 'mod'
        else:
            use_model = kwargs.pop('model')
        if 'fun_km' not in kwargs:
            fun_km = None
        else:
            fun_km = kwargs.pop('fun_km')
        if 'fun_e' not in kwargs:
            fun_e = None
        else:
            fun_e = kwargs.pop('fun_e')
        if 'model' not in kwargs:
            use_model = 'mod2'
        else:
            use_model = kwargs.pop('use_model')

        if nRsupp is None:
            nRsupp = np.sum(Rsupp > 0)

        y0 = np.concatenate([N0, R0])

        # solve ode, deal with different t format
        odefun = lambda t, y: self.model(t, y, dil=dil, Rsupp=Rsupp, nRsupp=nRsupp,
                                         fun_km=fun_km, fun_e=fun_e, use_model=use_model)
    
        if len(t) == 2:
            sol = solve_ivp(odefun, [t[0], t[-1]], y0, **kwargs)
        else:
            sol = solve_ivp(odefun, [t[0], t[-1]], y0, t_eval=t, **kwargs)

        # None-negative constrain, taking numerical accuracy into consideration
        thre_N = kwargs['atol']  # prohibit bounce back of extinct species by using numerical error
        thre_R = 0
        sol.y[:self.nS, :][sol.y[:self.nS, :] < thre_N] = 0
        sol.y[self.nS:, :][sol.y[self.nS:, :] < thre_R] = 0
        
        return sol


    # Simulate with daily dilution
    def sim_daily_dil(self, nday, dil, N0, R0, Rsupp=None, T=24, t=None, **kwargs):
        """
        dil_daily vs. dil_continuous: dil_d = 1 - (1 - dil_c)^(1/T)
        """
        # if kwargs doesn't specify atol and rtol, use default values
        if 'atol' not in kwargs:
            kwargs['atol'] = 1e-6
        if 'rtol' not in kwargs:
            kwargs['rtol'] = 1e-3

        if Rsupp is None:
            Rsupp = R0

        sol_list = np.empty((nday,), dtype=object)  # Preallocate array for solutions
        if t is None:
            t_evals = np.arange(nday).reshape([-1, 1]) * T + np.array([0, T]).reshape([1, -1])
        else:
            assert t[0] >= 0 and t[-1] <= T, "t must be within [0, T]"
            t_evals = np.arange(nday).reshape([-1, 1]) * T + np.array(t).reshape([1, -1])

        y0 = np.concatenate([N0, R0])
        for day in range(nday):
            # y0[y0 < 0] = 0  # set negative values (occur due to numerical accuracy limit) to 0. done in `sim`
            one_day_sol = self.sim(t_evals[day], y0[:self.nS], y0[self.nS:], dil=0, Rsupp=Rsupp, **kwargs)
            sol_list[day] = one_day_sol
            y0 = one_day_sol.y[:, -1] * 1
            y0[:self.nS] *= dil
            y0[self.nS:] = dil * y0[self.nS:] + (1 - dil) * Rsupp
        return MergedSolution(sol_list)

        pass

    # concat multiple models by "+" operation, class method
    def __add__(self, other):
        """add new species into the model, return a new model as an object of class ResourceModel
        the new species are described as an object of class ResourceModel
        new species are appended to the end of the original model

        crm1 = ConsumerResourceModel(nS1, nR, v0_1, km0_1, r0_1, e_1, b_1, gamma_1)
        crm2 = ConsumerResourceModel(nS2, nR, v0_2, km0_2, r0_2, e_2, b_2, gamma_2)
        crm = crm1 + crm2
        """
        assert self.nR == other.nR, "All models must have the same number of resources"
        cmb = ConsumerResourceModel(self.nS + other.nS, self.nR,
                                    v0=np.vstack([self.v0, other.v0]),
                                    km0=np.vstack([self.km0, other.km0]),
                                    r0=np.vstack([self.r0, other.r0]),
                                    e=np.vstack([self.e, other.e]),
                                    b=np.vstack([self.b, other.b]),
                                    gamma=np.vstack([self.gamma, other.gamma]))
        return cmb

    # concat multiple models, class method
    @classmethod
    def concat_models(cls, models):
        """concatenate multiple models, return a new model as an object of class ResourceModel
        """
        assert all([model.nR == models[0].nR for model in models]), "All models must have the same number of resources"
        nS = sum([model.nS for model in models])
        nR = models[0].nR
        v0 = np.vstack([model.v0 for model in models])
        km0 = np.vstack([model.km0 for model in models])
        r0 = np.vstack([model.r0 for model in models])
        e = np.vstack([model.e for model in models])
        b = np.vstack([model.b for model in models])
        gamma = np.vstack([model.gamma for model in models])
        cmb = ConsumerResourceModel(nS, nR, v0, km0, r0, e, b, gamma)
        return cmb

    def subset(self, species_idx=None, resource_idx=None):
        """subset the model with selected species and resources
        return a new model as an object of class ResourceModel
        """
        if species_idx is None:
            species_idx = np.arange(self.nS)
        if resource_idx is None:
            resource_idx = np.arange(self.nR)
        submod = ConsumerResourceModel(len(species_idx), len(resource_idx),
                                       v0=self.v0[species_idx, :][:, resource_idx],
                                       km0=self.km0[species_idx, :][:, resource_idx],
                                       r0=self.r0[species_idx],
                                       e=self.e[species_idx, :][:, resource_idx],
                                       b=self.b[species_idx, :][:, resource_idx, :][:, :, resource_idx],
                                       gamma=self.gamma[species_idx])
        return submod


# main function
if __name__ == "__main__":
    # generate model with random parameters
    pass
