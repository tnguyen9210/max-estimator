"""
trying to run bandits and see the performance of the estimator.
"""
module Tst

using Infiltrator

include("./KjBandits.jl")

#-------------------------------------------------------------------------------------

function problem_factory(problem_name::String, σ²::Real, seed::UInt32=987)
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
opt = (T=300, #1000
       σ² = (1.0)^2, 
       dataseed=123,
       algoseed=789,
       n_trials = 1000, # 200
       algo_name = "uniform", # "ucb", "sr", "uniform"
       problem_name = "linear_K20" #"linear", "linear_K2_custom", "linear_K2"
)
estimator_names = ["avg", "max", "weighted_100", "weightedms", "weightedms_4", "weightedms-gen"]
@show estimator_names
@show opt

hatvalues = zeros(opt.n_trials, opt.T, length(estimator_names))
global estimators
global i_t_ary
rng_data = MersenneTwister(opt.dataseed)
rng_algo = MersenneTwister(opt.algoseed) 
rng_algo2 = MersenneTwister(opt.algoseed*31 + 1)
algo = []
problem = []
for i_try = 1:opt.n_trials
    global algo, problem
    myseed = rand(rng_data, UInt32)
    problem = problem_factory(opt.problem_name, opt.σ², myseed)

    algo = algo_factory(opt.algo_name, problem, rand(rng_algo, UInt32))

    myseed2 = rand(rng_algo2, UInt32)
    estimators = [ estimator_factory(name, opt.σ², myseed2) for name in estimator_names]

    global i_t_ary
    i_t_ary = zeros(Int, opt.T)
    for t in 1:opt.T
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
me = meansqueeze(hatvalues, 1)
st = stdsqueeze(hatvalues, 1)
sterr = 2*st ./ sqrt(opt.n_trials)

using Plots
gr()

x = 1:opt.T
plot(x, ones(size(me)[1]) * .9, color=:black)
i = 1; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
i = 2; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
i = 3; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
i = 4; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
i = 5; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
i = 6; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])

# for i in eachindex(estimators)
#     if (i == 1)
#         nothing
#     else
#         plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
#     end
# end




@infiltrate


end
