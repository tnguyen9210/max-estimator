"""
trying to run bandits and see the performance of the estimator.

"""

using Infiltrator
using ProgressMeter
using Plots
plotly()

include("./KjBandits.jl")
include("./KjFunctions.jl")


opt = (
    σ² = (1.0/2)^2, 
    dataseed = 313, #123
    algoseed = 789,
    n_trials = 200, #50, #1000, #200
    N_ary = [10, 20, 40, 100, 1000, 2000, 4000, 6000, 8000, 10000], 
    algo_name = "uniform", # "ucb", # "ucb", "sr", "uniform"
    problem_name = "K_gap_poly", #"equal", "allbest", #"linear_K20" #"linear", "linear_K2_custom", "linear_K2"
    K = 64,
    num_optimals = nothing,
    degree = 2,
    Delta = nothing,
    gap_splits = nothing,
)
#estimator_names = ["haver", "naiveoracle", "avg", "max", "weighted_100", "weightedms_4", ]
estimator_names = ["naiveoracle", "haver", "avg", "max", "weightedms_4", "double"]
@show estimator_names
@show opt

hatvalues = zeros(opt.n_trials, length(opt.N_ary), length(estimator_names))
global estimators
global i_t_ary
rng_data = MersenneTwister(opt.dataseed)
rng_algo = MersenneTwister(opt.algoseed) 
rng_algo2 = MersenneTwister(opt.algoseed*31 + 1)
algo = []
problem = []
@showprogress for i_try = 1:opt.n_trials
#    global algo, problem
    myseed = rand(rng_data, UInt32)

    for i_N in eachindex(opt.N_ary)
        global problem, algo
        N = opt.N_ary[i_N]

        problem = problem_factory(
            opt.problem_name, opt.σ²,  myseed, 
            K=opt.K, num_optimals=opt.num_optimals, 
            degree=opt.degree, Delta=opt.Delta, gap_splits=opt.gap_splits)

        algo = algo_factory(opt.algo_name, problem, rand(rng_algo, UInt32))

        myseed2 = rand(rng_algo2, UInt32)
        estimators = [ estimator_factory(name, opt.σ², problem.μ, myseed2) for name in estimator_names]
    
        data = zeros(opt.K, N)
        for i in 1:opt.K
            algo.n_pulls[i] = N
            rewards = problem.μ[i] .+ sqrt(problem.σ²)*randn(problem.rngs[i], N) 
            algo.sum_rewards[i] = sum(rewards) 
            data[i,:] .= rewards
        end 
        
        for i_est in 1:length(estimators)-1
            hatvalues[i_try, i_N, i_est] = estimate_value(estimators[i_est], algo)
        end
        hatvalues[i_try, i_N, end] = estimate_value(estimators[end], data)
        
        @assert maximum(problem.μ) == 0.0
    end
    #- compute the bias and the MSE
end
bias = meansqueeze(hatvalues, 1) # n_alphas x n_estimators
mse = meansqueeze(hatvalues .^ 2, 1)
mae = meansqueeze(abs.(hatvalues), 1)

x = opt.N_ary
i = 1; plot(x, log.(mse[:,i]), label=estimator_names[i], linewidth=3)
i = 2; plot!(x, log.(mse[:,i]), label=estimator_names[i], linewidth=3)
# i = 3; plot!(x, log.(mse[:,i]), label=estimator_names[i], linewidth=3) 
i = 4; plot!(x, log.(mse[:,i]), label=estimator_names[i], linewidth=3)
i = 5; plot!(x, log.(mse[:,i]), label=estimator_names[i], linewidth=3)
i = 6; plot!(x, log.(mse[:,i]), label=estimator_names[i], linewidth=3)
plot!(xlabel="N", ylabel="log(MSE)", legend=:outertopright)  
title!("mse, K-gap, poly, degree=2")
png("mse_K_gap_poly_degree_2")

x = opt.N_ary
i = 1; plot(x, bias[:,i], label=estimator_names[i], linewidth=3)
i = 2; plot!(x, bias[:,i], label=estimator_names[i], linewidth=3)
# i = 3; plot!(x, bias[:,i], label=estimator_names[i], linewidth=3)
i = 4; plot!(x, bias[:,i], label=estimator_names[i], linewidth=3)
i = 5; plot!(x, bias[:,i], label=estimator_names[i], linewidth=3)
i = 6; plot!(x, bias[:,i], label=estimator_names[i], linewidth=3)
plot!(xlabel="N", ylabel="bias", legend=:outertopright)  
title!("bias, K-gap, poly, degree=2")
png("bias_K_gap_poly_degree_2")
