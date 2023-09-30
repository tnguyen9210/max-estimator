

import numpy as np
import scipy.stats as stats
np.set_printoptions(precision=4, suppress=4)

import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style("ticks")
sns.set_palette("tab10")
colors = sns.color_palette("tab10")

np.random.seed(123)

# params
num_trials = 10000
num_samples = 50
num_dims = 2
i_coord = 0

tau = 0.1
tau2 = tau**2
eps = tau2

catoni_delta = 0.05
catoni_nsearchs = 10

theta_star = np.array([1.0, 1.0]).reshape(2, 1)

x0 = np.array([1, -tau])
x1 = np.array([1, tau])
x2 = np.array([0, 1])

Q_inv_denom = tau**2*(8*tau**4 - 3*tau**2 - 1)
Q_inv_00 = tau**2*(tau**2 - 2)
Q_inv_01 = tau*(3*tau**2 - 1)
Q_inv_10 = tau*(3*tau**2 - 1)
Q_inv_11 = tau**2 - 1
Q_inv = np.array([[Q_inv_00, Q_inv_01], [Q_inv_10, Q_inv_11]])/Q_inv_denom
# print(f"Q = {Q}")
print(f"Q_inv = {Q_inv}")


def catoni_fn(samples_theta, catoni_alpha, est_theta):
    num_samples = len(samples_theta)
    samples_phi_x = catoni_alpha*(samples_theta - est_theta)
    samples_phi_x_pos_idxes = np.where(samples_phi_x >= 0)
    samples_phi_x_neg_idxes = np.where(samples_phi_x < 0)
    
    samples_phi_x2 = samples_phi_x**2
    samples_phi_val_pos = np.log(1 + samples_phi_x + samples_phi_x2/2.0)
    samples_phi_val_neg = -np.log(1 - samples_phi_x + samples_phi_x2/2.0)

    samples_phi_val = np.zeros(num_samples)
    samples_phi_val[samples_phi_x_pos_idxes] = \
        samples_phi_val_pos[samples_phi_x_pos_idxes]
    samples_phi_val[samples_phi_x_neg_idxes] = \
        samples_phi_val_neg[samples_phi_x_neg_idxes]
        
    # print(samples_phi_x[:5])
    # print(samples_phi_x_pos_idxes[:5])
    # print(samples_phi_val_pos[:5])
    # print(samples_phi_val_neg[:5])
    # print(samples_phi_val[:5])
    
    samples_R = np.sum(samples_phi_val)
    return samples_R

def catoni_estimator(samples_theta, catoni_var, catoni_delta, catoni_nsearchs):
    # params
    num_samples = len(samples_theta)

    # compute catoni_alpha
    nom = 2.0*np.log(1.0/catoni_delta)
    denom = num_samples*catoni_var*(1.0 + nom/(num_samples - nom))
    catoni_alpha = np.sqrt(nom/denom)

    # find the upper and lower bounds for the theta
    cur_theta = np.mean(samples_theta)
    cur_R = catoni_fn(samples_theta, catoni_alpha, cur_theta)
    # print(f"cur_theta = {cur_theta:0.4f}")
    # print(f"cur_R = {cur_R:0.4f}")
    
    eps = 0.01
    if cur_R >= 0:
        # print(f"pos cur_R")
        while cur_R >= 0:
            prev_theta = cur_theta
            cur_theta += eps
            cur_R = catoni_fn(samples_theta, catoni_alpha, cur_theta)
            # print(f"prev_theta = {prev_theta:0.4f}")
            # print(f"cur_theta = {cur_theta:0.4f}")
        lower_theta = prev_theta
        upper_theta = cur_theta
        
    elif cur_R < 0:
        # print(f"neg cur_R")
        while cur_R < 0:
            prev_theta = cur_theta
            cur_theta -= eps
            cur_R = catoni_fn(samples_theta, catoni_alpha, cur_theta)
            # print(f"prev_theta = {prev_theta:0.4f}")
            # print(f"cur_theta = {cur_theta:0.4f}")
        upper_theta = prev_theta
        lower_theta = cur_theta
        
    # double check whether we get the right upper and lower bounds
    upper_R = catoni_fn(samples_theta, catoni_alpha, upper_theta)
    lower_R = catoni_fn(samples_theta, catoni_alpha, lower_theta)
    assert upper_R < 0, "stop upper_R = {upper_R} >= 0"
    assert lower_R >= 0, "stop lower_R = {lower_R} < 0"

    acc_thres = 0.0001
    while (upper_theta - lower_theta) > acc_thres:
        # print(f"upper_theta = {upper_theta:0.4f}")
        # print(f"lower_theta = {lower_theta:0.4f}")
        cur_theta = (upper_theta + lower_theta)/2
        cur_R = catoni_fn(samples_theta, catoni_alpha, cur_theta)
        if cur_R >= 0:
            lower_theta = cur_theta
        elif cur_R < 0:
            upper_theta = cur_theta

    # print(f"upper_theta = {upper_theta:0.4f}")
    # print(f"lower_theta = {lower_theta:0.4f}")
    cur_theta = (upper_theta + lower_theta)/2
    # print(f"cur_theta = {cur_theta:0.4f}")
    return cur_theta

        
trials_cm_theta = []
trials_cp_theta = []
for i_trial in range(num_trials):
    samples_x_idxes = np.random.choice(3, size=(num_samples,), p=[1-2*eps, eps, eps])
    samples_x_idxes = np.random.choice(3, size=(num_samples,), p=[0.5, 0.3, 0.2])
    samples_x = np.zeros((num_samples, num_dims))
    samples_x[samples_x_idxes == 0,:] = x0
    samples_x[samples_x_idxes == 1,:] = x1
    samples_x[samples_x_idxes == 2,:] = x2

    # print(samples_x_idxes[:10])
    # print(samples_x[:10])
    
    # cnt_0 = np.sum(samples_x_idxes == 0)
    # cnt_1 = np.sum(samples_x_idxes == 1)
    # cnt_2 = np.sum(samples_x_idxes == 2)
    # print(cnt_0 / num_samples)
    # print(cnt_1 / num_samples)
    # print(cnt_2 / num_samples)

    noise = np.random.normal(size=(num_samples,1))
    samples_y = np.matmul(samples_x, theta_star) + noise

    # samples_xy = samples_x*samples_y
    samples_theta = np.matmul(samples_x*samples_y, Q_inv)
    
    # regular estimator (averaing) 
    cm_theta_i = np.mean(samples_theta[:, i_coord])
    trials_cm_theta.append(cm_theta_i)

    # catoni estimator
    catoni_var = ((1 + tau)**2 + 1)*Q_inv[i_coord, i_coord]
    cp_theta_i = catoni_estimator(
        samples_theta[:, i_coord], catoni_var, catoni_delta, catoni_nsearchs)
    trials_cp_theta.append(cp_theta_i)
    

fig, axes = plt.subplots(
        nrows=2, ncols=1, sharex=False, sharey=False, figsize=(8,8))
axes = axes.ravel()

# plot histogram
num_bins = 50
cm_theta_hist, cm_theta_bins = \
    np.histogram(trials_cm_theta, bins=num_bins, density=True)
cm_theta_range = (cm_theta_bins[:-1] + cm_theta_bins[1:]) / 2  # centerize

cp_theta_hist, cp_theta_bins = \
    np.histogram(trials_cp_theta, bins=num_bins, density=True)
cp_theta_range = (cp_theta_bins[:-1] + cp_theta_bins[1:]) / 2  # centerize

axes[0].bar(cm_theta_range, height=cm_theta_hist,
            width=(cm_theta_bins[1]-cm_theta_bins[0]),
            color='None', edgecolor=colors[0], linestyle='-', label="catoni minus")
axes[0].bar(cm_theta_range, height=cp_theta_hist,
            width=(cp_theta_bins[1]-cp_theta_bins[0]),
            color='None', edgecolor=colors[1], linestyle='--', label="catoni plus")
# ax.plot(cm_theta_range, cm_theta_hist, 'g--', label="estimated density")
# ax.plot(x, px, 'r--', label="true density")
axes[0].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

# plot cdf
cm_ecdf_fn = stats.ecdf(trials_cm_theta)
cp_ecdf_fn = stats.ecdf(trials_cp_theta)

cm_ecdf_fn.cdf.plot(axes[1], color=colors[0], label="catoni minus")
cp_ecdf_fn.cdf.plot(axes[1], color=colors[1], label="catoni plus")
axes[1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.show()


# plt.show()

# plt.hist(trials_cm_theta, cumulative=True, label='CDF DATA', 
#          histtype='step', alpha=0.55, color='purple')
# plt.hist(trials_cm_theta, cumulative=True, label='CDF DATA', 
#          histtype='step', alpha=0.55, color='orange')
# plt.show()

# sns.displot(data=trials_cm_theta, kind="ecdf")
# sns.displot(data=trials_cp_theta, kind="ecdf")
# ax.ecdf(trials_cm_theta, label="new")
# plt.show()
