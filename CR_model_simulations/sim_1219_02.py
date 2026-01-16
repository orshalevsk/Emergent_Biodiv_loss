import copy
import numpy as np
import pickle
from consumer_resource import ConsumerResourceModel
import concurrent.futures
import datetime
import os


def generate_crmodel(nS, nR, seeds, rho_gamma=0, e_mean=0.5, **kwargs):
    """Generate a consumer-resource model with random parameters

    Parameters
    ----------
    nS : int
        Number of species
    nR : int
        Number of resources
    seeds : int or list of int, optional
        Random seed(s) to initialize the random number generator. If not provided, the random number generator will be initialized with the current system time.
    rho_gamma: float in [-1, 1]
        correlation between gamma and number of resources consumed

    Returns
    -------
    crmodel : ConsumerResourceModel
        A consumer-resource model with random parameters
    """
    crmodel_list = []

    for seed in seeds:
        np.random.seed(seed)
        # number of resources consumed by each species
        nR_consumed = np.random.randint(1, nR + 1, nS)
        v0 = np.zeros((nS, nR))
        for i in range(nS):
            consumed_indices = np.random.choice(nR, nR_consumed[i], replace=False)
            v0[i, consumed_indices] = np.random.uniform(1, 1, nR_consumed[i])  # v=1 for consumed resources for simplicity
        km0 = np.random.uniform(0, 1, (nS, nR))
        r0 = np.random.uniform(100, 100, nS)
        e = np.random.randn(nS, nR) * 0.1 + e_mean
        e[e < 0] = 0
        b = np.random.uniform(-2, 1, (nS, nR, nR))
        b[b < 0] = 0  # sparse
        b = b / (np.sum(b, axis=2, keepdims=True) + 1e-10)
        # gamma correlated with nR_consumed with correlation coefficient rho_gamma
        # the gamma values here needs to be latter processed to be binarized to 0 and gamma_hacc
        gamma = rho_gamma * (nR_consumed - np.mean(nR_consumed)) / np.std(nR_consumed)  + np.random.randn(nS) * np.sqrt(1 - rho_gamma ** 2)
        paras = {'v0': v0, 'km0': km0, 'r0': r0, 'e': e, 'b': b, 'gamma': gamma}

        # process kwargs to replace random parameters
        for key, value in kwargs.items():
            if key in ['v0', 'km0', 'r0', 'e', 'b', 'gamma']:
                paras[key] = value
            else:
                print(f'Warning from generate_crmodel(): {key} is not a valid key. Ignored.')
        crmodel = ConsumerResourceModel(nS, nR, **paras)
        crmodel_list.append(crmodel)

    return crmodel_list

def sim_one_comm(crm_paras, list_nRsupp, seed=None, long_return=False):
    """
    simulate one community for different numbers of supplied resources
    """
    crm = ConsumerResourceModel(**crm_paras)
    nS = crm.nS
    nR = crm.nR
    sol_list = np.zeros(len(list_nRsupp), dtype=object)
    N0 = np.ones(nS) / nS

    if long_return:
        sol_list = []
    if seed is not None:
        np.random.seed(seed)
    final_abundance = np.zeros((len(list_nRsupp), nS + nR))
    for idiv, nRsupp in enumerate(list_nRsupp):
        # Build resource supply vector
        R0 = np.zeros(crm.nR)
        R0[np.random.choice(crm.nR, nRsupp, replace=False)] = 1 / nRsupp

        # Simulate model. mod2 - growth not capped by resource uptake
        sol = crm.sim(np.array([0, 10, 50, 100, 200, 400, 800]), N0, R0, dil=0.1, Rsupp=R0, atol=1e-6, model='mod2')

        if long_return:
            sol_list.append(sol)
        final_abundance[idiv, :] = sol.y[:, -1]

    if long_return:
        return final_abundance, sol_list
    else:
        return final_abundance

def general_smoothstep(x, x0, k, y_max, y_min=0):
    """
    A general cubic smoothstep. Used to map the rawly generated gamma_0 to gamma_hacc and gamma_macc or a smooth transition.

    Parameters:
    - x: Input values
    - f: The horizontal shift (the point where y = h/2)
    - k: The slope at point f
    - h: The maximum height (y-range is [0, h])
    """
    # Calculate the total width of the transition based on desired slope k
    # In 3x^2 - 2x^3, the max slope is 1.5.
    # To scale slope to 'k' and height to 'h', the width must be:
    h = y_max - y_min
    width = (1.5 * h) / k

    # Normalize x to the [0, 1] range relative to the window around f
    # x_norm will be 0.5 when x = x0
    x_norm = (x - x0) / width + 0.5
    x_norm = np.clip(x_norm, 0, 1)  # Clamp to [0, 1] to ensure the "step" stays at 0 and h

    # The cubic smoothstep formula
    return y_min + h * (3 * x_norm**2 - 2 * x_norm**3)


# nohup nice -n 10 python3 -u sim_1219_02.py > simres_1219_02/log_nohup 2>&1 &
if __name__ == "__main__":
    # save path
    res_folder_all = "./simres_1219_02"

    # Model parameters
    nS, nR = 20, 30
    nRsupp_list = np.repeat([1, 2, 4, 8, 16], 3)

    # Fractions of HACC to test
    f_list = np.concatenate([np.arange(0, 0.3, 0.05), np.arange(0.3, 1.01, 0.1)])
    comm_list = np.arange(20)
    nComm = len(comm_list) # Number of communities per parameter set

    e_mean = 0.4
    rho_gamma = 0.4
    gamma_macc, gamma_hacc = 0, 1
    k_gamma_list = [0.4, 0.8, 1.6, 3, 6, 12]  # k_gamma==np.inf corresponds to step function as in default settings

    for k_gamma in k_gamma_list:
        res_folder = f"{res_folder_all}/kgamma={k_gamma:.1f}"
        try:
            os.mkdir(res_folder)
        except FileExistsError as e:
            print(e, flush=True)
            print("skipped simulations", flush=True)
            continue

        print("*****************************************************************************")
        print(f"***Starting simulations for k_gamma={k_gamma}***", flush=True)
        print("*****************************************************************************")

        crm_pool = generate_crmodel(1000, nR, range(1), rho_gamma=rho_gamma, e_mean=e_mean)[0]  # same crm pool for each (rho_gamma, e_mean)

        # parallel computing using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_idx = {}  # future -> (ii, ifrac, x)
            futures_per_comm = {ii: [None] * len(f_list) for ii in
                                comm_list}  # each entry stores futures for that ii (community)
            pending_per_comm = {ii: len(f_list) for ii in comm_list}  # number of pending futures per community
            finish_comm_count = 0
            crm_base_dict = {}  # store base crm for each community

            # submit all jobs
            for ii in comm_list:  # for each community
                np.random.seed(ii * 41)  # same species selection for each (rho_gamma, e_mean)
                spe_idx = np.random.choice(1000, nS, replace=False)
                crm_base = crm_pool.subset(spe_idx)
                crm_base = crm_base.subset(np.argsort(crm_base.gamma.flatten())) # sorted from low to high gamma
                crm_base_dict[ii] = crm_base  # store base crm
                for ifrac, f in enumerate(f_list):  # for each highly impacted strain fraction f
                    # Number of HACC and MACC species. MACC first, HACC follows.
                    n_hacc = int(nS * f)
                    n_macc = nS - n_hacc

                    crm_curr = copy.deepcopy(crm_base)
                    if k_gamma < 100: # map from raw gamma to smoothstep-transformed gamma based on x0 and k (slope)
                        if n_hacc == 0:  # step after the last species (biggest gamma)
                            x0 = crm_curr.gamma[-1] + (crm_curr.gamma[-1] - crm_curr.gamma[-2])  # extrapolate
                        elif n_macc == 0:  # step before the first species (smallest gamma)
                            x0 = crm_curr.gamma[0] - (crm_curr.gamma[1] - crm_curr.gamma[0])  # extrapolate
                        else: # step between the two species around the cutoff
                            x0 = (crm_curr.gamma[n_macc - 1] + crm_curr.gamma[n_macc]) / 2
                        crm_curr.gamma = general_smoothstep(crm_curr.gamma, x0=x0, k=k_gamma,
                                                            y_max=gamma_hacc, y_min=gamma_macc)
                    else:  # step function
                        crm_curr.gamma = np.concatenate([np.ones(n_macc) * gamma_macc, np.ones(n_hacc) * gamma_hacc])

                    # Submit simulation to executor
                    fut = executor.submit(sim_one_comm, crm_curr.get_parameters(), nRsupp_list,
                                          seed=ii * 17, long_return=True)  # same resource selection for each (frac, rho_gamma, e_mean)
                    future_to_idx[fut] = (ii, ifrac)  # map future to its indices
                    futures_per_comm[ii][ifrac] = fut  # store futures
            print(f"{datetime.datetime.now()}: Submitted all simulations.", flush=True)

            # process and save community by community
            for fut in concurrent.futures.as_completed(future_to_idx):
                ii2, ifrac2 = future_to_idx[fut]
                pending_per_comm[ii2] -= 1
                print(
                    f"{datetime.datetime.now()}: Community {ii2 + 1}/{nComm}, Fraction {f_list[ifrac2]:.2f} done. Pending for this community: {pending_per_comm[ii2]}", flush=True)

                # if all futures for this community are done, process and save
                if pending_per_comm[ii2] == 0:
                    print(f'com{ii2} - checkpoint 1', flush=True)
                    finish_comm_count += 1
                    final_abundance_allfracs = [None] * len(f_list)

                    fail_count = 0
                    for ifrac3 in range(len(f_list)):
                        try:
                            final_abundance_allfracs[ifrac3] = futures_per_comm[ii2][ifrac3].result()
                        except Exception as e:
                            print(f"Error in Community {ii2}, Fraction {f_list[ifrac3]:.2f}: {e}", flush=True)
                            final_abundance_allfracs[ifrac3] = None
                            fail_count += 1
                    if fail_count > len(f_list) * 0.2:
                        raise RuntimeError(f"Too many failed simulations for Community {ii2}. Aborting program.", flush=True)

                    # save to file
                    print(f'com{ii2} - checkpoint 2', flush=True)
                    save_filename = f"{res_folder}/Nfin_com{ii2:03d}.pkl"
                    with open(save_filename, 'wb') as file:
                        pickle.dump({
                            'final_abundance_allfracs': final_abundance_allfracs,
                            'nRsupp_list': nRsupp_list,
                            'f_list': f_list,
                            'crm_base': crm_base_dict[ii2],
                            'gamma_macc': gamma_macc,
                            'gamma_hacc': gamma_hacc,
                            'rho_gamma': rho_gamma,
                            'e_mean': e_mean,
                            'k_gamma': k_gamma,
                        }, file)
                    print(f"\n==== Simulation Done for {finish_comm_count}/{nComm} Communities ====", flush=True)
                    # remove stored data to save memory
                    del futures_per_comm[ii2]