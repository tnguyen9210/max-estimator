"""
trying to run bandits and see the performance of the estimator.

let's see what happens as the number of samples gets larger when the true mean is all equal.
"""
module Tst

using Infiltrator

include("./KjBandits.jl")
using ProgressMeter

#-------------------------------------------------------------------------------------

# function problem_factory(problem_name::String, σ²::Real, seed::UInt32=987)
#     if     problem_name == "linear"
#         problem = Bandit(collect(0.9:-0.1:0.0), σ², seed)
#     elseif     problem_name == "linear_K100"
#         problem = Bandit(collect(LinRange(1,0,100)), σ², seed)
#     elseif     problem_name == "linear_K20"
#         problem = Bandit(collect(LinRange(1,0,20)), σ², seed)
#     elseif problem_name == "linear_K2"
#         problem = Bandit([0.9,0.5], σ², seed)
#     elseif problem_name == "linear_K2_custom"
#         problem = Bandit([0.9,0.3], σ², seed)
#     end
#     problem
# end
function problem_factory(problem_name::String, σ²::Real, seed::UInt32=987; K=-1, alpha=1.0, Delta=0.1)
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
        Delta_ary = ((1:20) ./ 20) .^ alpha
        mu = -Delta_ary
        mu[1] = 0.0
        problem = Bandit(mu, σ², seed)
    elseif problem_name == "equal_K20"
        Delta_ary = Delta*ones(20)
        mu = -Delta_ary
        mu[1] = 0.0
        problem = Bandit(mu, σ², seed)
    elseif problem_name == "equal"
#        Delta_ary = (.1*2)*ones(K)
        Delta_ary = (.1*16)*ones(K)
        mu = -Delta_ary
        mu[1] = 0.0
        problem = Bandit(mu, σ², seed)
    elseif problem_name == "allbest"
        mu = zeros(K)
        problem = Bandit(mu, σ², seed)
    end
    problem
end

function estimator_factory(name, σ², mu=Float64[], seed=123)
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
    elseif name == "naiveoracle"
        est = NaiveOracle(mu)
    elseif name == "haver"
        est = Haver(σ²,0.05)
    else
        @error "value error"
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
opt = (T_max = 20*50,
       K=10,
       σ² = (1.0)^2, 
       dataseed=123,
       algoseed=789,
       n_trials = 1000, #1000, # 200
       algo_name = "uniform", # "ucb", "sr", "uniform"
       problem_name = "equal", #"allbest", #"linear_K20" #"linear", "linear_K2_custom", "linear_K2"
)
#estimator_names = ["naiveoracle", "avg", "max", "weighted_100", "weightedms_4"]
estimator_names = ["naiveoracle", "avg", "max", "weightedms_4", "haver"]
@show estimator_names
@show opt

hatvalues = zeros(opt.n_trials, opt.T_max, length(estimator_names))
global estimators
global i_t_ary
rng_data = MersenneTwister(opt.dataseed)
rng_algo = MersenneTwister(opt.algoseed) 
rng_algo2 = MersenneTwister(opt.algoseed*31 + 1)
algo = []
problem = []
@showprogress for i_try = 1:opt.n_trials
    global algo, problem
    myseed = rand(rng_data, UInt32)
    problem = problem_factory(opt.problem_name, opt.σ², myseed, K=opt.K)

    algo = algo_factory(opt.algo_name, problem, rand(rng_algo, UInt32))

    myseed2 = rand(rng_algo2, UInt32)
    estimators = [ estimator_factory(name, opt.σ², myseed2) for name in estimator_names]

    global i_t_ary
    i_t_ary = zeros(Int, opt.T_max)
    for t in 1:opt.T_max
        i_t = next_arm(algo)
        reward = pull(problem, i_t)
        update!(algo, i_t, reward)

        for i in eachindex(estimators)
            hatvalues[i_try, t, i] = estimate_value(estimators[i], algo)
        end

        i_t_ary[t] = i_t
    end

    cumreg = cumsum(calc_inst_regret(problem, i_t_ary))
end
bias = meansqueeze(hatvalues, 1) # n_alphas x n_estimators
mse = meansqueeze(hatvalues .^ 2, 1)

using Plots
gr()

# x = opt.Deltas
# i = 1; plot(x, bias[:,i], label=estimator_names[i])
# i = 2; plot!(x, bias[:,i], label=estimator_names[i])
# i = 3; plot!(x, bias[:,i], label=estimator_names[i])
# i = 4; plot!(x, bias[:,i], label=estimator_names[i])
# i = 5; plot!(x, bias[:,i], label=estimator_names[i])
# i = 6; plot!(x, bias[:,i], label=estimator_names[i])
# title!("bias")

x = problem.K:opt.T_max
i = 1; plot(log.(x), log.(mse[x,i]), label=estimator_names[i],linewidth=3)
i = 2; plot!(log.(x), log.(mse[x,i]), label=estimator_names[i],linewidth=3)
i = 3; plot!(log.(x), log.(mse[x,i]), label=estimator_names[i],linewidth=3) 
i = 4; plot!(log.(x), log.(mse[x,i]), label=estimator_names[i],linewidth=3)
i = 5; plot!(log.(x), log.(mse[x,i]), label=estimator_names[i],linewidth=3, xtickfont=font(18), ytickfont=font(18), xlabel="log(n_samples)", ylabel="log(MSE)", labelfontsize=18,legend=:bottomleft)  
title!("mse")




# i = 1; plot!(x, me[x,i], ribbon=(sterr[x,i], sterr[x,i]), label=estimator_names[i])
# i = 2; plot!(x, me[x,i], ribbon=(sterr[x,i], sterr[x,i]), label=estimator_names[i])
# i = 3; plot!(x, me[x,i], ribbon=(sterr[x,i], sterr[x,i]), label=estimator_names[i])
# i = 4; plot!(x, me[x,i], ribbon=(sterr[x,i], sterr[x,i]), label=estimator_names[i])
# 
# for i in eachindex(estimators)
#     if (i == 1)
#         nothing
#     else
#         plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
#     end
# end




@infiltrate


end

