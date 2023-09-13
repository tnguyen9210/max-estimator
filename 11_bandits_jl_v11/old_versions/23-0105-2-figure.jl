"""
trying to run bandits and see the performance of the estimator.

but do it in a way where the X axis is varying Delta
"""
module Tst

using Infiltrator
using LinearAlgebra

include("./KjBandits.jl")

#-------------------------------------------------------------------------------------

function problem_factory(problem_name::String, σ²::Real, alpha=1.0, Delta=0.1, seed::UInt32=987)
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
    elseif problem_name == "equal_tmp"
        Delta_ary = Delta*ones(10)
        mu = -Delta_ary
        mu[1] = 0.0
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
opt = (
       σ² = (1.0)^2, 
       dataseed=123,
       algoseed=789,
       n_trials = 1000, # 200
       algo_name = "uniform", # "ucb", "sr", "uniform"
       problem_name = "equal_tmp", # "poly_K20", "linear_K20", "linear", "linear_K2_custom", "linear_K2"
       Deltas = LinRange(0.1, 1, 10),
)
#estimator_names = ["avg", "max", "weighted_100", "weightedms", "weightedms_4", "weightedms-gen"]
#estimator_names = ["max", "weighted_100", "weightedms", "weightedms_4", "naiveoracle", "avg"]
estimator_names = ["max", "weighted_100", "weightedms_4", "weightedms", "naiveoracle"]
@show estimator_names
@show opt

hatvalues = zeros(opt.n_trials, length(opt.Deltas), length(estimator_names))
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

    for i_Delta in eachindex(opt.Deltas)
        global problem, algo
        Delta = opt.Deltas[i_Delta]

        problem = problem_factory(opt.problem_name, opt.σ², NaN, Delta, myseed)

        algo = algo_factory(opt.algo_name, problem, rand(rng_algo, UInt32))

        myseed2 = rand(rng_algo2, UInt32)
        estimators = [ estimator_factory(name, opt.σ², problem.μ, myseed2) for name in estimator_names]

        T = problem.K * 2
        for t in 1:T
            i_t = next_arm(algo)
            reward = pull(problem, i_t)
            update!(algo, i_t, reward)
        end

        for i in eachindex(estimators)
            hatvalues[i_try, i_Delta, i] = estimate_value(estimators[i], algo)
        end
        @assert maximum(problem.μ) == 0.0
    end
    #- compute the bias and the MSE
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

x = opt.Deltas
i = 1; plot(x, mse[:,i], label="max",linewidth=3)
i = 2; plot!(x, mse[:,i], label=estimator_names[i],linewidth=3)
i = 3; plot!(x, mse[:,i], label=estimator_names[i],linewidth=3) 
i = 4; plot!(x, mse[:,i], label=estimator_names[i],linewidth=3)
i = 5; plot!(x, mse[:,i], label=estimator_names[i],linewidth=3, xtickfont=font(18), ytickfont=font(18), legendfont=font(18), xlabel="Delta", labelfontsize=18) 



# for i in eachindex(estimators)
#     if (i == 1)
#         nothing
#     else
#         plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
#     end
# end




@infiltrate


end



