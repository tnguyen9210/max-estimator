"""
trying to run bandits and see the performance of the estimator.

but do it in a way where the X axis is varying the problem parameter, rather than the number of iterations.
"""
module Tst

using Infiltrator
using LinearAlgebra

include("./KjBandits.jl")

#-------------------------------------------------------------------------------------

function problem_factory(problem_name::String, σ²::Real, alpha=1.0, seed::UInt32=987)
    if     problem_name == "linear"
        problem = Bandit(collect(0.9:-0.1:0.0), σ², seed)
    elseif     problem_name == "linear_K100"
        problem = Bandit(collect(LinRange(1,0,100)), σ², seed)
    elseif     problem_name == "linear_K20"
        problem = Bandit(collect(LinRange(1,0,20)), σ², seed)
    elseif problem_name == "linear_K2"
        problem = Bandit([0.9,0.5], σ², seed)
    elseif problem_name == "linear_K2_custom"
        problem = Bandit([0.9,0.3], σ², seed)
    elseif problem_name == "poly_K20"
        Delta = ((1:20) ./ 20) .^ alpha
        mu = -Delta
        mu[1] = 0.0
        problem = Bandit(mu, σ², seed)
    end
    problem
end

function estimator_factory(name, σ², seed=123)
    if (name == "avg")
        est = Average()
    elseif name == "max"
        est = Max()
    elseif name == "weighted_100"
        est = Weighted(σ², 100, seed)
    elseif name == "weightedms"
        est = WeightedMs(σ²)
    elseif name == "weightedms-gen"
        est = WeightedMsGen(σ²,3)
    elseif name == "weightedms_4"
        est = WeightedMs(σ²,4)
    end
    est
end

function algo_factory(algo_name::String, problem::Bandit, seed)
    if algo_name == "ucb"
        algo = Ucb(problem.K, problem.σ², seed=seed)
    elseif algo_name == "sr"
        algo = Rejects(problem.K, seed)
    elseif algo_name == "uniform"
        algo = Uniform(problem.K, seed)
    end
    algo
end

#-------------------------------------------------------------------------------------
opt = (T=40, #1000          
       σ² = (1.0)^2, 
       dataseed=123,
       algoseed=789,
       n_trials = 1000, # 200
       algo_name = "uniform", # "ucb", "sr", "uniform"
       problem_name = "poly_K20", # linear_K20" #"linear", "linear_K2_custom", "linear_K2"
       alphas = LinRange(.1,2, 10)
)
estimator_names = ["avg", "max", "weighted_100", "weightedms", "weightedms_4", "weightedms-gen"]
@show estimator_names
@show opt

hatvalues = zeros(opt.n_trials, length(opt.alphas), length(estimator_names))
global estimators
global i_t_ary
rng_data = MersenneTwister(opt.dataseed)
rng_algo = MersenneTwister(opt.algoseed) 
rng_algo2 = MersenneTwister(opt.algoseed*31 + 1)
algo = []
problem = []
for i_try = 1:opt.n_trials
#    global algo, problem
    myseed = rand(rng_data, UInt32)

    for i_alpha in eachindex(opt.alphas)
        global problem, algo
        alpha = opt.alphas[i_alpha]

        problem = problem_factory(opt.problem_name, opt.σ², alpha, myseed)

        algo = algo_factory(opt.algo_name, problem, rand(rng_algo, UInt32))

        myseed2 = rand(rng_algo2, UInt32)
        estimators = [ estimator_factory(name, opt.σ², myseed2) for name in estimator_names]

        for t in 1:opt.T
            i_t = next_arm(algo)
            reward = pull(problem, i_t)
            update!(algo, i_t, reward)
        end

        for i in eachindex(estimators)
            hatvalues[i_try, i_alpha, i] = estimate_value(estimators[i], algo)
        end
        @assert maximum(problem.μ) == 0.0
    end
    #- compute the bias and the MSE
end
bias = meansqueeze(hatvalues, 1) # n_alphas x n_estimators
mse = meansqueeze(hatvalues .^ 2, 1)

using Plots
gr()

x = opt.alphas
i = 1; plot(x, bias[:,i], label=estimator_names[i])
i = 2; plot!(x, bias[:,i], label=estimator_names[i])
i = 3; plot!(x, bias[:,i], label=estimator_names[i])
i = 4; plot!(x, bias[:,i], label=estimator_names[i])
i = 5; plot!(x, bias[:,i], label=estimator_names[i])
i = 6; plot!(x, bias[:,i], label=estimator_names[i])
title!("bias")

i = 1; plot(x, mse[:,i], label=estimator_names[i])
i = 2; plot!(x, mse[:,i], label=estimator_names[i])
i = 3; plot!(x, mse[:,i], label=estimator_names[i])
i = 4; plot!(x, mse[:,i], label=estimator_names[i])
i = 5; plot!(x, mse[:,i], label=estimator_names[i])
i = 6; plot!(x, mse[:,i], label=estimator_names[i])
title!("mse")


# for i in eachindex(estimators)
#     if (i == 1)
#         nothing
#     else
#         plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
#     end
# end




@infiltrate


end

