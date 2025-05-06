import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import solve_ivp
import pickle
from scipy.interpolate import interp1d
import datetime
import warnings

# warnings.filterwarnings("ignore", message="xxx")

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

    Hypothesis:
    defalut para: diversity increases with resrouce diversity
    HACC vs. MACC: HACC takes up more resources as resource diversity increases
    changing HACC/MACC distribution should change species div vs res div relationship

    Key features for base model:
    * spe i uptake res j with a rate of v_ij = v0_ij * R_j / (km_ij + R_j), and turn it into biomass with an efficiency of e_ij
        for simplicity, default to e_ij = 1 (no cross-feeding), v_ij = 1
    * if e_ij < 1, the rest energy (1 - e_ij) * v_ij is secreted as other resources
        b_ijk (S * R * R matrix) specifying species i secretes the unusable part of resource j to resource k(s)
            sum(b_ijk, axes=2) = 1
    * diff spe takes up diff res, modify affility by km, maybe also by a 1or0 v_ij matrix
    * spe i has a max growth rate of r0_i
    * realized growth rate of spe i is a_i = sum_j(e_ij * v_ij), if > r0_i scale down to r0_i
        i.e. r_i = min(a_i, r0_i)
    * the resource uptake is c_ij = v_ij * r_i / a_i
        this assumes that the spe can adjust its uptake rate proportionally for each resource to match its growth rate
    * dN_i/dt = r_i * N_i
    * dR_j/dt = - uptake + secretion
        uptake_j = sum_i(c_ij * N_i)
        secretion_k = sum_ij(b_ijk * (1 - e_ij) * v_ij * N_i)
    * dilution is either continuous or daily

    Key features for HACC vs MACC:
    simple version (preferable, although less supported by data):
    * km_ij is a function of res diversity, km_ij = km0_ij * (1 + f(gamma_i, R_div))
    complex version:
    * km_ij is a function of res identity, km_ij = max_j(km0_ij for all R_j > 0) or something like this

    More thoughts:
    * v_ij and a_ij also play a huge role in bacteria's response to resource vector. How to set a base line needs careful consideration
    * Desired behavior:
        * HACC: low R_div -> high Km, more left
    * In a continuous / daily dilution scenario, things can be quite different
        * in continous:
        * try with example

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

    # Define the differential equations
    def model(self, t, y, dil, Rsupp, nRsupp=1, **kwargs):
        """Define the ode-model for the consumer-resource system

        """
        N, R = copy.deepcopy(y[:self.nS]), copy.deepcopy(y[self.nS:])
        # treat negative values (occur due to numerical accuracy limit) as 0
        N[N < 0] = 0
        N = N.reshape([-1, 1])  # nS * 1
        R[R < 0] = 0
        R = R.reshape([1, -1])  # 1 * nR

        # TODO: test different ways that gamma affects km
        #km = self.km0 / (1 + self.gamma * np.sum(R > 1e-1))  # nS * nR
        #km = self.km0 / (1 + self.gamma * np.sum(R > 0.0001))           
        if nRsupp < 2:
            # No gamma effect
            km = self.km0
        else:
            # Normal gamma-based formula
            km = self.km0 / (1 + self.gamma * np.sum(R > 0.0001))
        km = np.maximum(km, 1e-6)  # avoid numerical error
        # ----- TESTING BEGIN -----
        if 'fun_km' in kwargs:  # lambda gamma, R: f(gamma, R). note input is of size nR * 1
            km = self.km0 * kwargs['fun_km'](self.gamma, R)
        # ----- TESTING END -----
        v = self.v0 * R / (km + R)  # nS * nR

        # TODO: does the scale of a make sense?
        a = np.sum(self.e * v, axis=1, keepdims=True)  # nS * 1, realized growth rate. maximum equals number of resource
        r = np.minimum(a, self.r0)
        dNdt = r * N - dil * N
        c = v * r / a  # nS * nR, consumption
        uptake = np.sum(c * N, axis=0, keepdims=True)  # nR * 1
        secretion = np.sum(np.sum(self.b * ((1 - self.e) * v * N)[:, :, None], axis=0), axis=0, keepdims=True) # nR * 1. sum over spe then res
        dRdt = -uptake + secretion - dil * R + dil * Rsupp
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
        t can be [tini, tend] or a list of time points to evaluate the solution
        """
        # if kwargs doesn't specify atol and rtol, use default values
        if 'atol' not in kwargs:
            kwargs['atol'] = 1e-6
        if 'rtol' not in kwargs:
            kwargs['rtol'] = 1e-3

        # if kwargs doesn't specify method, use RK45
        if 'method' not in kwargs:
            kwargs['method'] = 'RK45'

        y0 = np.concatenate([N0, R0])

        # solve ode, deal with different t format
        odefun = lambda t, y: self.model(t, y, dil=dil, Rsupp=Rsupp, nRsupp=nRsupp, **kwargs)
    
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


def generate_crmodel(nS, nR, seeds, **kwargs):
    """Generate a consumer-resource model with random parameters

    Parameters
    ----------
    nS : int
        Number of species
    nR : int
        Number of resources
    seeds : int or list of int, optional
        Random seed(s) to initialize the random number generator. If not provided, the random number generator will be initialized with the current system time.

    Returns
    -------
    crmodel : ConsumerResourceModel
        A consumer-resource model with random parameters
    """
    crmodel_list = []

    for seed in seeds:
        np.random.seed(seed)
        v0 = np.random.uniform(1, 1, (nS, nR))
        km0 = np.random.uniform(0, 10, (nS, nR))
        r0 = np.random.uniform(0.2, 1, nS)
        e = np.random.uniform(1, 1, (nS, nR))
        b = np.random.uniform(-3, 1, (nS, nR, nR))
        b[b < 0] = 0  # sparse
        b = b / np.sum(b, axis=2, keepdims=True)
        gamma = np.random.uniform(0, 0, nS)

        crmodel = ConsumerResourceModel(nS, nR, v0, km0, r0, e, b, gamma, **kwargs)
        crmodel_list.append(crmodel)

    return crmodel_list

# main function
if __name__ == "__main__":
    # generate model with random parameters
    pass
