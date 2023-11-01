
def weightedms2_q_learning(
        env, num_actions, num_steps_train,
        gamma, lr_sched_fn, eps_sched_fn, tdqm_disable, args=None):

    num_depths = args["num_depths"]
    num_actions_half = int(num_actions/2)
    action_sigma = args["action_sigma"]
    
    # keep track of useful statistics
    stats = []
    num_data = args["weightedms_num_data"]
    num_actions = env.action_space.n
    Q_table = defaultdict(lambda: inf_Q*np.ones(num_actions))
    Q2_table = defaultdict(lambda: inf_Q*np.ones(num_actions))
    Q_nvisits = defaultdict(lambda: np.zeros(num_actions))

    Q_sigmahats = defaultdict(lambda: np.ones(num_actions)*1e10)
    Q_ess = defaultdict(lambda: 0*np.ones(num_actions))
    Q_ess_weights = defaultdict(lambda: 0*np.ones(num_actions))
    Q2_ess_weights = defaultdict(lambda: 0*np.ones(num_actions))
    
    cur_state, info = env.reset()
    cur_state = f"{cur_state}"
    start_state = copy.deepcopy(cur_state)
    
    for i_step in tqdm(
            range(num_steps_train), desc="train q_learning", disable=tdqm_disable):
        
        nvisits = np.sum(Q_nvisits[cur_state]) + 1
        # eps = 1.0/np.sqrt(nvisits)
        eps = eps_sched_fn(nvisits)
        action = eps_greedy_policy(
            Q_table[cur_state], Q_nvisits[cur_state], cur_state, num_depths, eps)

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = f"{new_state}"
        # print(f"\n-> i_step = {i_step}")
        # print(f"cur_state = {cur_state}")
        # print(f"action = {action}")
        # print(f"reward = {reward:0.2f}")
        # print(f"new_state = {new_state}")
        # if terminated:
        #     print("terminated")

        if not terminated:
            
            elems = re.findall(r'\d+', new_state)
            new_state_depth = int(elems[0])
            new_state_width = int(elems[1])

            if new_state_depth == num_depths-1:
                Q_est = Q_table[new_state][0]
                
            elif new_state_width < num_actions_half:
                action_muhats = copy.deepcopy(Q_table[new_state][:num_actions_half])
                action_nvisits = Q_nvisits[new_state][:num_actions_half]
                action_muhats[action_nvisits == 0] = -inf_mu
                
                action_sigmahats = Q_sigmahats[new_state][:num_actions_half]
                action_sigmahats[action_sigmahats < 1e-5] = 1e-5

                Q_est = weightedms2_estimator(
                    action_muhats, action_sigmahats, action_nvisits,
                    num_actions_half, num_data)

            elif new_state_width >= num_actions_half:
                action_muhats = copy.deepcopy(Q_table[new_state][num_actions_half:])
                action_nvisits = Q_nvisits[new_state][num_actions_half:]
                action_muhats[action_nvisits == 0] = -inf_mu
                
                action_sigmahats = Q_sigmahats[new_state][num_actions_half:]
                action_sigmahats[action_sigmahats < 1e-5] = 1e-5

                Q_est = weightedms2_estimator(
                    action_muhats, action_sigmahats, action_nvisits, 
                    num_actions_half, num_data)

            # print(f"Q_table[new_state] = {Q_table[new_state]}")
            # print(f"Q_est[new_state] = {Q_est:0.2f}")
            
            # compute td_target
            td_target = reward + gamma*Q_est
        else:
            td_target = reward

        # print(f"td_error = {td_target-Q_table[cur_state][action]:0.2f}")
        # print(f"Q_nvisits[cur_state][action] = {Q_nvisits[cur_state][action]+1}")
        # print(f"Q_table[cur_state][action], before = {Q_table[cur_state][action]:0.2f}")
        Q_nvisits[cur_state][action] += 1
        lr = lr_sched_fn(Q_nvisits[cur_state][action])
        
        Q_table[cur_state][action] = \
            (1-lr)*Q_table[cur_state][action] + lr*td_target
        Q2_table[cur_state][action] = \
            (1-lr)*Q2_table[cur_state][action] + lr*td_target**2
        # print(f"Q_table[cur_state][action], after = {Q_table[cur_state][action]:0.2f}")

        if Q_nvisits[cur_state][action] >= 1:
            Q_ess_weights[cur_state][action] = \
                (1-lr)*Q_ess_weights[cur_state][action] + lr
            Q2_ess_weights[cur_state][action] = \
                (1-lr)**2*Q2_ess_weights[cur_state][action] + lr**2
            Q_ess[cur_state][action] = \
                1./Q2_ess_weights[cur_state][action]

        if Q_nvisits[cur_state][action] >= 1:
            diff = Q2_table[cur_state][action] - Q_table[cur_state][action]**2
            diff = action_sigma**2
            if diff < 0:
                diff = 0
            Q_sigmahats[cur_state][action] = np.sqrt(diff/Q_ess[cur_state][action])

        if terminated:
            cur_state, info = env.reset()
            cur_state = f"{cur_state}"
        else:
            cur_state = new_state

        action_muhats = copy.deepcopy(Q_table[start_state])
        action_nvisits = Q_nvisits[start_state]
        action_muhats[action_nvisits == 0] = -inf_mu
        
        action_sigmahats = Q_sigmahats[start_state]
        action_sigmahats[action_sigmahats < 1e-5] = 1e-5
        
        Q_start_est = weightedms2_estimator(
            action_muhats, action_sigmahats, action_nvisits, num_actions, num_data)
        
        stats.append((np.sum(Q_nvisits[start_state]),
                      reward, Q_start_est))

    Q_table_ary, Q_nvisits_ary = \
        convert_tables(Q_table, Q_nvisits, num_actions, env.num_depths)

    return Q_table_ary, Q_nvisits_ary, stats
    # return Q_table, stats

def weightedms2_estimator(
        action_muhats, action_sigmahats, action_nvisits, num_actions, num_data):
    # cur_muhats = Q_table[est_state]
    # cur_sigmahats = Q_sigmahats[est_state]
    # cur_sigmahats[cur_sigmahats < 1e-4] = 1e-4

    if np.sum(action_nvisits) == 0:
        return 0
    
    # sample 
    # action_muhats_mat = matlib.repmat(action_muhats, num_data, 1)
    # action_sigmahats_mat = matlib.repmat(action_sigmahats, num_data, 1)
    # eps_mat = np.random.randn(num_data, num_actions)
    # samples = action_muhats_mat + action_sigmahats_mat*eps_mat
    samples = np.random.normal(
        action_muhats, action_sigmahats, size=(num_data, num_actions))
    
    # samples[action_nvisits == 0,:] = -inf_mu
    samples_max_idxes = np.argmax(samples, 1)

    # compute probs 
    probs = np.zeros(num_actions)
    idxes, cnts = np.unique(samples_max_idxes, return_counts=True)
    probs[idxes[cnts > 0]] = cnts[cnts > 0]
    probs = probs/num_data

    # print(action_sigmahats)
    # print(action_muhats)
    # print(probs)
    
    # compute Q_est 
    Q_est = np.dot(
        probs[action_nvisits != 0], action_muhats[action_nvisits != 0])
    
    # Q_est = Q_est if np.sum(action_nvisits) != 0 else 0.0
    # print(f"Q_est_ = {Q_est:0.2f}")
    # stop
    return Q_est

