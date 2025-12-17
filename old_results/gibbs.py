# from tqdm import tqdm 
# import numpy as np
# import scipy.stats as stats
# from densities import unnormalized_posterior_mu_logpdf, q_tilde_logpdf, psi, psi_inverse, fy_pdf


# def update_mu_metropolis(mu_current, x_current, params):
#     """
#     Performs one Metropolis-Hastings step to get a new sample for mu.
#     """
#     # 1. Propose a new candidate for mu from a symmetric proposal distribution.
#     #    We use a Normal distribution centered at the current mu.
#     mu_candidate = np.random.normal(loc=mu_current, scale=params['proposal_std_mu'])

#     # 2. Calculate the log-posterior for the current and candidate mu values.
#     log_post_current = unnormalized_posterior_mu_logpdf(
#         mu=mu_current,
#         x=x_current,
#         params=params
#     )
#     log_post_candidate = unnormalized_posterior_mu_logpdf(
#         mu=mu_candidate,
#         x=x_current,
#         params=params
#     )

#     # 3. Calculate the acceptance probability (alpha) in log-space.
#     #    alpha = min(1, p(candidate)/p(current))
#     #    log(alpha) = min(0, log(p(candidate)) - log(p(current)))
#     log_acceptance_ratio = log_post_candidate - log_post_current
#     acceptance_prob = min(1.0, np.exp(log_acceptance_ratio))

#     # 4. Accept or reject the candidate.
#     if np.random.uniform(0, 1) < acceptance_prob:
#         return mu_candidate, True
#     else:
#         return mu_current, False
    
    
# def update_z(z_current, delta, mu_current, mu_star, params):
#     """
#     Performs one Metropolis-Hastings step to get a new sample for z_i.
#     (This function's logic remains unchanged, but it now benefits from the
#      more stable q_tilde_logpdf function.)

#     Returns:
#         (float, bool): A tuple containing:
#                        - The new z_i value.
#                        - A boolean indicating if the proposal was accepted.
#     """
#     # 1. Propose a new candidate for z, ensuring it's in the valid domain.
#     k = params['k']
#     supp_q = (-1/(2*np.sqrt(k)), 1/(2*np.sqrt(k)))
    

#     proposal_std_z = params['proposal_std_z']
#     supp_tilde_q = (max(supp_q[0], delta - supp_q[1]), min(supp_q[1], delta - supp_q[0]))
#     kernel = params.get('kernel', 'normal')
#     cpt_try = 0
#     while True:
#         cpt_try += 1
#         if kernel == "normal":
#             z_candidate = np.random.normal(loc=z_current, scale=proposal_std_z)
#             log_ratio_kernel = 0.0  # Symmetric proposal
            
#             if supp_tilde_q[0] < z_candidate < supp_tilde_q[1]:
#                 break
#         elif kernel == "truncated_normal":
#             a_candidate, b_candidate = (supp_tilde_q[0] - z_current) / proposal_std_z, (supp_tilde_q[1] - z_current) / proposal_std_z
            
#             z_candidate = stats.truncnorm.rvs(a_candidate, b_candidate, loc=z_current, scale=proposal_std_z)
            
#             a_current, b_current = (supp_tilde_q[0] - z_candidate) / proposal_std_z, (supp_tilde_q[1] - z_candidate) / proposal_std_z
            
#             log_ratio_kernel = stats.truncnorm.logpdf(z_current, a_current, b_current, loc=z_candidate, scale=proposal_std_z) - stats.truncnorm.logpdf(z_candidate, a_candidate, b_candidate, loc=z_current, scale=proposal_std_z)
#             break
#         elif kernel == "uniform":
#             z_candidate = np.random.uniform(low=supp_tilde_q[0], high=supp_tilde_q[1])
#             log_ratio_kernel = 0.0  # Symmetric proposal
#             break
#         else:
#             raise ValueError(f"Unknown kernel type: {kernel}")

#     # 2. Calculate the log-posterior (q_tilde_logpdf) for current and candidate z.
#     log_post_current = q_tilde_logpdf(z_current, delta, mu_current, mu_star, k)
#     log_post_candidate = q_tilde_logpdf(z_candidate, delta, mu_current, mu_star, k)

#     # 3. Calculate the acceptance probability.
#     log_acceptance_ratio = log_post_candidate - log_post_current + log_ratio_kernel
#     if np.isneginf(log_acceptance_ratio):
#         acceptance_prob = 0.0
#     else:
#         acceptance_prob = min(1.0, np.exp(log_acceptance_ratio))

#     # 4. Accept or reject the candidate.
#     if np.random.uniform(0, 1) < acceptance_prob:
#         return z_candidate, True, cpt_try
#     else:
#         return z_current, False, cpt_try
    
# def update_xi_xj(xi, xj, mu_current, mu_star, params):
#     """
#     Performs the full 6-step constrained pairwise update for indices (i, j).

#     Returns:
#         (np.array, bool, bool): A tuple containing:
#                                 - The new data vector.
#                                 - A boolean for whether the PAIR was updated.
#                                 - A boolean for whether the Z_I proposal was accepted.
#     """
#     k = params['k']
#     supp_q = (-1/(2*np.sqrt(k)), 1/(2*np.sqrt(k)))
#     # --- Step 1: Calculate delta ---
#     yi, yj = xi - mu_star, xj - mu_star
#     zi, zj = psi(yi, k), psi(yj, k)
#     delta = zi + zj

#     # --- Step 2: Sample z_tilde_i ---
#     zi_tilde, zi_accepted, cpt_try = update_z(
#         z_current=zi, delta=delta, mu_current=mu_current, mu_star=mu_star, params=params
#     )
#     # print(f"Update z_i: current={zi:.4f}, proposed={zi_tilde:.4f}, accepted={zi_accepted}")

#     # --- Step 3: Set Partner and Check Domain ---
#     zj_tilde = delta - zi_tilde
    
#     if not (supp_q[0] < zj_tilde < supp_q[1]):
#         print("NOT IN SUPPORT")
#         return (xi, xj), False, zi_accepted, cpt_try

#     # --- Step 4: Compute inverse branches ---
#     yi_minus, yi_plus = psi_inverse(zi_tilde, k)
#     yj_minus, yj_plus = psi_inverse(zj_tilde, k)

#     # --- Step 5: Assign weights to each of the 4 pairs ---

#     # create two arrays.
#     # One for all the y_i candidates and one for all the y_j candidates.
#     ys_candidate = np.array([yi_minus, yi_plus, yj_minus, yj_plus])
#     weights = fy_pdf(ys_candidate, mu_current, mu_star, k)
    
#     weights = np.array([
#         weights[0] * weights[2],  # (y_i_minus, y_j_minus)
#         weights[0] * weights[3],  # (y_i_minus, y_j_plus)
#         weights[1] * weights[2],  # (y_i_plus, y_j_minus)
#         weights[1] * weights[3],  # (y_i_plus, y_j_plus)
#     ])
    
#     # --- Step 6: Sample a pair  ---
#     sum_of_weights = np.sum(weights)
#     if sum_of_weights <= 0 or np.isnan(sum_of_weights):
#         return (xi, xj), False, zi_accepted, cpt_try

#     probs = weights / sum_of_weights

#     # We need the original list of tuples to select the winner from
#     candidate_y_pairs = [(yi_minus, yj_minus), (yi_minus, yj_plus), (yi_plus, yj_minus), (yi_plus, yj_plus)]
    
#     chosen_index = np.random.choice(4, p=probs)
#     y_i_new, y_j_new = candidate_y_pairs[chosen_index]

#     x_i_new = y_i_new + mu_star
#     x_j_new = y_j_new + mu_star
    
#     return (x_i_new, x_j_new), True, zi_accepted, cpt_try



# def update_x_full(x_current, mu_current, mu_star, params):
#     """
#     Performs a full systematic scan to update the entire x vector.

#     Returns:
#         (np.array, int, int): A tuple containing:
#                               - The new, fully updated data vector.
#                               - The number of accepted PAIRS.
#                               - The number of accepted Z_I proposals.
#     """
#     x_to_update = np.copy(x_current)
#     m = params['m']
#     pair_accepted_count = 0
#     z_i_accepted_count = 0
#     cpt_try_total = 0

#     x_to_update = np.random.permutation(x_to_update)
#     for k in range(params['num_z_moves']):
#         i, j = 2*k, 2*k + 1
#         xi, xj = x_to_update[i], x_to_update[j]
#         indices = (i, j)
#         (x_i_new, x_j_new), pair_accepted, z_i_accepted, cpt_try = update_xi_xj(xi, xj, mu_current, mu_star, params
#         )
#         cpt_try_total += cpt_try
#         if pair_accepted:
#             pair_accepted_count += 1
#         if z_i_accepted:
#             z_i_accepted_count += 1
#         # print(f"Before xi, xj = {xi:.4f}, {xj:.4f}  x to update[i,j] = {x_to_update[i]:.4f}, {x_to_update[j]:.4f}")
#         x_to_update[i], x_to_update[j] = x_i_new, x_j_new
#         # print(f"After  xi, xj = {x_i_new:.4f}, {x_j_new:.4f}  x to update[i,j] = {x_to_update[i]:.4f}, {x_to_update[j]:.4f}")

#     return x_to_update, pair_accepted_count, z_i_accepted_count, cpt_try_total
    
    
# def run_gibbs_sampler_mle(mu_star, params):
#     T = params['num_iterations_T']
#     m = params['m']
#     mus = np.zeros(T+1)
#     xs = np.zeros((T+1, m))
    
#     x_0 = np.ones(m) * mu_star
#     xs[0, :] = x_0
    
#     mu_0 = mu_star
#     mus[0] = mu_0
    
#     num_z_moves = params.get('num_z_moves', m//2)
#     params['num_z_moves'] = num_z_moves
#     x_current = x_0.copy() # Use a copy to avoid modifying the original
#     mu_acceptance_count = 0
#     z_i_acceptance_count = 0
#     pair_acceptance_count = 0
#     cpt_try_total = 0
    
#     total_z_moves = T * num_z_moves 

#     # The main loop
#     for t in tqdm(range(1, T+1), desc="Running Gibbs Sampler"):
#         # --- Step (a): Sample mu(t) ---
#         mus[t], accept_mu = update_mu_metropolis(mus[t-1], x_current, params)
#         if accept_mu: 
#             mu_acceptance_count += 1
            
#         # --- Step (b): Sample x(t) ---
#         xs[t], accepted_pairs, accepted_z_is, cpt_try = update_x_full(xs[t-1], mus[t], mu_star, params)

#         z_i_acceptance_count += accepted_z_is
#         pair_acceptance_count += accepted_pairs
#         cpt_try_total += cpt_try


#     # Calculate final rates
#     mu_acceptance_rate = mu_acceptance_count / T
#     z_i_acceptance_rate = z_i_acceptance_count / total_z_moves 
#     cpt_try_rate = cpt_try_total / total_z_moves
#     pair_acceptance_rate = pair_acceptance_count / total_z_moves

#     print(f"\n--- Sampling Complete ---")
#     print(f"Mu Acceptance Rate: {mu_acceptance_rate:.4f}")
#     print(f"Z_i Acceptance Rate: {z_i_acceptance_rate:.4f}")
    
#     results = {
#         "mu_acceptance_rate": mu_acceptance_rate,
#         "pair_acceptance_rate": pair_acceptance_rate,
#         "z_i_acceptance_rate": z_i_acceptance_rate,
#         "cpt_try_rate": cpt_try_rate,
#         "mu_chain": mus,
#         "x_chain": xs,
#     }
    
#     return results





# from utils import get_mle


# def run_gibbs_sample_x(x, params):
#     T = params['num_iterations_T']
#     m = len(x)
#     mus = np.zeros(T+1)
    
#     mu_star = get_mle(x, params)
#     mus[0] = mu_star
    
#     mu_acceptance_count = 0
    
#     # The main loop
#     for t in tqdm(range(1, T+1), desc="Running Gibbs Sampler"):
#         # --- Step (a): Sample mu(t) ---
#         mus[t], accept_mu = update_mu_metropolis(mus[t-1], x, params)
#         if accept_mu: 
#             mu_acceptance_count += 1



#     # Calculate final rates
#     mu_acceptance_rate = mu_acceptance_count / T
    
#     print(f"\n--- Sampling Complete ---")
#     print(f"Mu Acceptance Rate: {mu_acceptance_rate:.4f}")
    
#     results = {
#         "mu_acceptance_rate": mu_acceptance_rate,
#         "mu_chain": mus,
#     }
    
#     return results
