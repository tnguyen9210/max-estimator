"""
the tree example
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
opt = (σ² = (1/4)^2, 
       dataseed=87,
       algoseed=789,
       n_trials = 50,
       T = 10_000,
)
# estimator_names = ["avg", "max", "weighted_100", "weightedms", "weightedms_4", "de", ]
# @show estimator_names

function gen_alg(name, c_uct, σ², tree, seed)
    alg = nothing
    if name == "uct"
        alg = Uct(tree, c_uct, σ², :average, seed=seed) 
    elseif name == "uct-wems"
        alg = Uct(tree, c_uct, σ², :wems, seed=seed)
    elseif name == "uct-wems-half"
        alg = Uct(tree, c_uct, σ², :wems_half, seed=seed)
    elseif name == "luct-wems"
        alg = Luct(tree, c_uct, σ², :wems, seed=seed)
    elseif name == "srt-wems"
        alg = Srt(tree, σ², seed=seed)
    else
        @error "name is incorrect"
    end
    alg
end

#- I tuned parameter from here (T=15_000)
# depth = 3
# width = 8

#- an okay setting
# depth = 2
# width = 15

depth = 2
width = 5

rnd_data = MersenneTwister(opt.dataseed)
rnd_algo = MersenneTwister(opt.algoseed)
c_uct = 1.0
alg_name_ary = ["uct", "uct-wems", "srt-wems"] # "uct", "uct-wems", "uct-wems-half", "srt-wems"
t_list = collect(20:20:opt.T)

b_first = true
gaps = zeros(opt.n_trials, length(alg_name_ary), length(t_list))
ts = time()
for i_trial in 1:opt.n_trials
    tree_seed = randseed(rnd_data)

    alg_seed = randseed(rnd_algo)
    for i_alg in eachindex(alg_name_ary)
        global b_first
        alg_name = alg_name_ary[i_alg]

        #tree = gen_toy_minmax_tree(depth, width, 1.0, UctData, seed=tree_seed)
        tree = gen_toy_maxmax_tree(depth, width, 1.0, UctData, seed=tree_seed)
#         if (b_first)
#             print_node(tree)
#             b_first = false
#         end

        alg = gen_alg(alg_name, c_uct, opt.σ², tree, alg_seed)

        for t in 1:opt.T
            leaf_t = next_leaf(alg)
            reward = pull(tree, leaf_t)
            update!(alg, leaf_t, reward)

            if (t in t_list)
                mu = [tree.root.children[i].data.value for i in 1:length(tree.root.children)]
                counts = [tree.root.children[i].data.t for i in 1:length(tree.root.children)] 
                hat_mu = [tree.root.children[i].data.hat_mu for i in 1:length(tree.root.children)] 
                 maxcnt = maximum(counts)
                 ties = findall(counts .== maxcnt)

                #--- how to recommend the arm?
#                  i_best = ties[argmax(hat_mu[ties])]
                i_best = argmax(hat_mu)
                
                gap = maximum(mu) - mu[i_best]
                gaps[i_trial,i_alg,findfirst(t .== t_list)] = gap
                if (i_trial == 2 && t == opt.T)
                    nothing
                    print_children_data(tree.root)
                    println("")
                    #@infiltrate
                    nothing
                end
            end
        end
    end
end
@show time() - ts

res = meansqueeze(gaps,1) # n_alg by length(t_list)

println("mean:")
display(res)
println("std:")
display(stdsqueeze(gaps,1))

using Plots
gr()

plot(t_list, res[1,:],label=alg_name_ary[1])
plot!(t_list, res[2,:],label=alg_name_ary[2])
plot!(t_list, res[3,:],label=alg_name_ary[3])



# hatvalues = zeros(opt.n_try, length(K_list), length(estimator_names))
# estimators = []
# rng_data = MersenneTwister(opt.dataseed)
# rng_algo2 = MersenneTwister(opt.algoseed*31 + 1)
# 
# for i_try = 1:opt.n_try
# 
#     for i_K in eachindex(K_list)
#         myseed = rand(rng_data, UInt32)
#         K = K_list[i_K]
#         problem = gen_my_bandit(K, Delta, myseed)
# 
#         myseed2 = rand(rng_algo2, UInt32)
#         estimators = [ estimator_factory(name, opt.σ², myseed2) for name in estimator_names]
# 
#         # n_pulls = convert(Int,round(K^(0.25)))*ones(Int, K) 
#         # n_pulls = n_pulls_each*ones(Int, K) 
#         
#         # avgs = [mean(gen_rewards(problem, i, n_pulls[i])) for i in 1:K]
#         
#         # n_pulls_each = 4
#         n_pulls_each = 2*convert(Int,ceil(K^(.25)))
#         data = zeros(K, n_pulls_each)
#         for i in 1:K
#             data[i,:] .= gen_rewards(problem, i, n_pulls_each)
#         end
# 
#         for i in eachindex(estimators)
# #            hatvalues[i_try, i_K, i] = estimate_value(estimators[i], avgs, n_pulls)
#             hatvalues[i_try, i_K, i] = estimate_value(estimators[i], data)
#         end
#     end
# end
# me = meansqueeze(hatvalues, 1)
# st = stdsqueeze(hatvalues, 1)
# sterr = 2*st ./ sqrt(opt.n_try)
# 
# 
# using Plots
# gr()
# 
# x = log2.(K_list)
# plot(x, ones(size(me)[1]) * 1.0, color=:black)
# i = 1; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
# i = 2; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
# i = 3; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
# i = 4; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
# i = 5; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i])
# i = 6; plot!(x, me[:,i], ribbon=(sterr[:,i], sterr[:,i]), label=estimator_names[i], legend=:topright, foreground_color_legend = nothing, background_color_legend=nothing)



@infiltrate


end

