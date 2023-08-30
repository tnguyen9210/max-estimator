"""
performance of WE-MS. 
run it in VSCode.

NEXT:
 - 
"""

using Random
using Statistics
n_trials = 10_000

Delta_list = 0:0.05:5

# FIXME: define sigma outside.
function f_we(v, sigma)
    hDelta = maximum(v) .- v
    X = sigma * randn(100,length(v)) .- hDelta'
    argmax_ary = argmax.(eachrow(X))

    w = [count(==(element),argmax_ary) for element in eachindex(v)]
    w /= sum(w)
    w'v
end
function f(v,sigma)
    hDelta = maximum(v) .- v
    w = @. exp(-hDelta^2/sigma^2)
    w /= sum(w)
    w'v
end
function f3(v) # forget about this one..
    hDelta = maximum(v) .- v
    w = @. (1 + hDelta)
    w /= sum(w)
    w'v
end
function f_me(v)
    maximum(v)
end
function f_oracle_smart(v, Delta, sigma)
    w = (Delta^2 + sigma^2) / (Delta^2 + 2*sigma^2)
    v[1] * w + v[2]*(1-w)
end
function f_avg(v)
    mean(v)
end
function f_hybrid(v, sigma)
    w0 = 1/sigma^2*ones(length(v))     # this is the same as average estimator!!
    hDelta = maximum(v) .- v
    x = @. sqrt(hDelta^2/(2*(sigma^2 + sigma^2)))
    w = @. w0 * min(1.0,1.0/x)*exp(-x^2)
    w /= sum(w)
    w'v
end

algo_names = ["wems-cheat", "wems", "oracle", "oracle-smart", "me", "we", "hybrid"] #, "avg"]
n_algo = length(algo_names)
bias = zeros(length(Delta_list),n_algo)
mse = zeros(length(Delta_list),n_algo)
sigma = 1.0 

Z = zeros(n_algo, n_trials)
for i in eachindex(Delta_list)
    global Z 
    Delta = Delta_list[i]
    X = sigma*randn(n_trials,2) .+ [0,-Delta]'

    Z[1,:] = @. (X[:,1] + X[:,2] * exp(-Delta^2/sigma^2)) / (1 + exp(-Delta^2/sigma^2))

    Z[2,:] = [f(r,sigma) for r in eachrow(X)]
    Z[3,:] = X[:,1]

    #- oracle
    Z[4,:] = [f_oracle_smart(r,Delta,sigma) for r in eachrow(X)]

    Z[5,:] = [f_me(r) for r in eachrow(X)]

    Z[6,:] = [f_we(r, sigma) for r in eachrow(X)]

    #Z[7,:] = [f_avg(r) for r in eachrow(X)]

    Z[7,:] = [f_hybrid(r, sigma) for r in eachrow(X)]

    bias[i,:] .= mean(Z, dims=2) #Z.mean(dims=1) [mean(Z1), mean(Z2), mean(Z3), mean(Z4), mean(Z5), mean(Z6)]
    mse[i,:] .= mean(Z.^2, dims=2) #[mean(Z1.^2), mean(Z2.^2), mean(Z3.^2), mean(Z4.^2), mean(Z5.^2), mean(Z6.^2)  ]
# 	@show mean(Z1)
# 	@show mean(Z2)
end

using Plots
for i in eachindex(algo_names)
    aname = algo_names[i]
    if (i == 1)
        plot(Delta_list, bias[:,i], label=aname)
    elseif i == length(algo_names)
        display(plot!(Delta_list, bias[:,i], label=aname))
    else
        plot!(Delta_list, bias[:,i], label=aname)
    end
end
for i in eachindex(algo_names)
    aname = algo_names[i]
    if (i == 1)
        plot(Delta_list, mse[:,i], label=aname)
    elseif i == length(algo_names)
        display(plot!(Delta_list, mse[:,i], label=aname))
    else
        plot!(Delta_list, mse[:,i], label=aname, legend=:bottomright)
    end
end