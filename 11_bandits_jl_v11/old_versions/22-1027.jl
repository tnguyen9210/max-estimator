"""
trying to run bandits and see the performance of the estimator.
"""
module Tst

using Infiltrator

include("./KjBandits.jl")

#-------------------------------------------------------------------------------------

function problem_factory(problem_name::String, σ²::Real, seed::UInt32=987)
    if     problem_name == "linear"
        problem = Bandit(collect(0.9:-0.1:0.1), σ², seed)
    elseif problem_name == "linear_K2"
        problem = Bandit([0.9,0.5], σ², seed)
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
    elseif name == "weightedms_4"
        est = WeightedMs(σ²,4)
    elseif name == "de"
        est = Double(seed)
    end
end

#-------------------------------------------------------------------------------------
opt = (σ² = 1^2, 
       dataseed=123,
       algoseed=789,
       n_try = 100,
       K_list = 2 .^(1:8), 
       Delta = 0.5, 
)
estimator_names = ["avg", "max", "weighted_100", "weightedms", "weightedms_4", "de", ]
@show estimator_names

K_list = opt.K_list
Delta = opt.Delta

function gen_my_bandit(K, Delta, seed)
    mu = (1-Delta)*ones(K)
    mu[1] = 1.0
    problem = Bandit(mu, opt.σ², seed)
end



hatvalues = zeros(opt.n_try, length(K_list), length(estimator_names))
estimators = []
rng_data = MersenneTwister(opt.dataseed)
rng_algo2 = MersenneTwister(opt.algoseed*31 + 1)

for i_try = 1:opt.n_try

    for i_K in eachindex(K_list)
        myseed = rand(rng_data, UInt32)
        K = K_list[i_K]
        problem = gen_my_bandit(K, Delta, myseed)

        myseed2 = rand(rng_algo2, UInt32)
        estimators = [ estimator_factory(name, opt.σ², myseed2) for name in estimator_names]

        # n_pulls = convert(Int,round(K^(0.25)))*ones(Int, K) 
        # n_pulls = n_pulls_each*ones(Int, K) 
        
        # avgs = [mean(gen_rewards(problem, i, n_pulls[i])) for i in 1:K]
        
        # n_pulls_each = 4
        n_pulls_each = 2*convert(Int,ceil(K^(.25)))
        data = zeros(K, n_pulls_each)
        for i in 1:K
            data[i,:] .= gen_rewards(problem, i, n_pulls_each)
        end

        for i in eachindex(estimators)
#            hatvalues[i_try, i_K, i] = estimate_value(estimators[i], avgs, n_pulls)
            hatvalues[i_try, i_K, i] = estimate_value(estimators[i], data)
        end
    end
end
me = meansqueeze(hatvalues, 1)
st = stdsqueeze(hatvalues, 1)
sterr = 2*st ./ sqrt(opt.n_try)


using Plots
gr()

x = log2.(K_list)
plot(x, ones(size(me)[1]) * 1.0, color=:black)
i = 1; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
i = 2; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
i = 3; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
i = 4; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
i = 5; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
i = 6; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i], legend=:topright, foreground_color_legend = nothing, background_color_legend=nothing)

################################################################################

# 
# hatvalues = zeros(opt.n_try, opt.T, length(estimator_names))
# estimators = []
# rng_data = MersenneTwister(opt.dataseed)
# rng_algo = MersenneTwister(opt.algoseed) 
# rng_algo2 = MersenneTwister(opt.algoseed*31 + 1)
# for i_try = 1:opt.n_try
#     myseed = rand(rng_data, UInt32)
#     problem = problem_factory("linear_K2", opt.σ², myseed)
#     algo = Ucb(problem.K, problem.σ², rand(rng_algo, UInt32))
# 
#     myseed2 = rand(rng_algo2, UInt32)
#     estimators = [ estimator_factory(name, opt.σ², myseed2) for name in estimator_names]
# 
#     i_t_ary = zeros(Int, opt.T)
#     for t in 1:opt.T
#         i_t = next_arm(algo)
#         reward = pull(problem, i_t)
#         update!(algo, i_t, reward)
# 
#         for i in eachindex(estimators)
#             hatvalues[i_try, t, i] = estimate_value(estimators[i], algo)
#         end
# 
#         i_t_ary[t] = i_t
#     end
# 
#     cumreg = cumsum(calc_inst_regret(problem, i_t_ary))
# end
# me = meansqueeze(hatvalues, 1)
# st = stdsqueeze(hatvalues, 1)
# sterr = 2*st ./ sqrt(opt.n_try)
# 




@infiltrate


end

