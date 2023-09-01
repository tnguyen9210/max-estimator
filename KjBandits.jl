using Infiltrator
using Random
using Statistics
using Printf
# include("KjUtils.jl")
# using .KjUtils

sumsqueeze(A, dims) = dropdims(sum(A, dims=dims); dims=dims)
meansqueeze(A, dims) = dropdims(mean(A, dims=dims); dims=dims)
stdsqueeze(A, dims) = dropdims(std(A, dims=dims); dims=dims)

################################################################################
# bandit problems
################################################################################

#---- goal: let's test on depth one tree = bandits
abstract type Problem end

struct Bandit <: Problem
    K::Int
    μ::Vector{Float64}
    σ²::Float64
    seed::Int
    rngs::Vector{MersenneTwister}

    Bandit(μ, σ², seed=987) = begin
        K = length(μ)

        rngs = [MersenneTwister(seed + i) for i in 1:K]
        new(K, μ, σ², seed, rngs) 
    end
end

function calc_inst_regret(problem::Problem, i_t_ary)
    max_mu = maximum(problem.μ)
    Deltas = max_mu .- problem.μ
    Deltas[i_t_ary]
end

function pull(self::Bandit, i::Int)
    self.μ[i] + sqrt(self.σ²)*randn(self.rngs[i])
end

function gen_rewards(self::Bandit, i::Int, n::Int)
    self.μ[i] .+ sqrt(self.σ²)*randn(self.rngs[i], n)
end

# struct BanditBernoulli <: Problem
#     K::Int
#     μ::Vector{Float64}
#     seed::Int
#     rngs::Vector{MersenneTwister}
# 
#     Bandit(μ, seed=987) = begin
#         K = length(μ)
# 
#         rngs = [MersenneTwister(seed + i) for i in 1:K]
#         new(K, μ, seed, rngs) 
#     end
# end
# 
# function pull(self::Bandit, i::Int)
#     convert(Float64, rand(self.rngs[i]) < self.mu[i])
# end




################################################################################
# bandit algorithms
################################################################################

abstract type BanditAlg end

mutable struct Ucb <: BanditAlg  
    const K::Int64
    const σ²::Float64
    const seed::Int64
    const rng::Random.MersenneTwister
    const sum_rewards::Vector{Float64} 
    const n_pulls::Vector{Int64} 
    const reverse::Bool
    const c::Float64 # scale the width
    t::Int64
end

function Ucb(K, σ²; reverse::Bool=false, seed=123, c=1.0)
    sum_rewards = zeros(K)
    n_pulls = zeros(K)
    rng = MersenneTwister(seed)

    Ucb(K, σ², seed, rng, sum_rewards, n_pulls, reverse, c, 1)
end

function next_arm(self::Ucb)
    me = self.sum_rewards ./ self.n_pulls

    width = self.c*sqrt.(2*self.σ² .* log(self.t * (1 + log(self.t)^2)) ./ self.n_pulls)
    # TODO ideally, need to break ties..!!
    if self.reverse
        lcbs = me .- width
        ret = argmin(lcbs)
    else
        ucbs = me .+ width
        ret = argmax(ucbs)
    end
    ret
end

function update!(self::BanditAlg, i_pulled, reward)
    # update pulls and then confidence bounds
    self.t += 1
    self.n_pulls[i_pulled] += 1
    self.sum_rewards[i_pulled] += reward
    nothing
end

"""
this is to be used in UCT!!
11/23: didn't I give up on this..?
"""
mutable struct Ucb2 <: BanditAlg
    const K::Int64
    const σ²::Float64
    const seed::Int64
    const rng::Random.MersenneTwister
#    const sum_rewards::Vector{Float64} 
    const n_pulls::Vector{Int64} 
    const hatmu::Vector{Float64}
    const vars::Vector{Float64}
    const reverse::Bool
    const c::Float64 # scale the width
    t::Int64
    
    # TODO make the constructor.
end

function next_arm(self::Ucb2)
    width = self.c*sqrt.(2*self.σ² .* log(self.t * (1 + log(self.t)^2)) ./ self.n_pulls)

    self.hatmu + width # could also use the variance information
    # TODO ideally, need to break ties..!!
    if self.reverse
        lcbs = self.hatmu .- width
        ret = argmin(lcbs)
    else
        ucbs = self.hatmu .+ width
        ret = argmax(ucbs)
    end
    ret
end

# TODO
"""
TODO: this should be updating the hatmu and the variance of the pulled arm!
"""
function update!(self::Ucb2, i_pulled, hatmu, var)
    # update pulls and then confidence bounds
    self.t += 1
    self.n_pulls[i_pulled] += 1
    #    self.sum_rewards[i_pulled] += reward # not needed

    self.hatmu[i_pulled] = hatmu
    self.vars[i_pulled] = vars
end

mutable struct Rejects <: BanditAlg  
    const K::Int64
    const seed::Int64
    const rng::Random.MersenneTwister
    const sum_rewards::Vector{Float64} 
    const n_pulls::Vector{Int64} 
    const p_star::Vector{Float64}
    t::Int64
end

function Rejects(K, seed=123)
    sum_rewards = zeros(K)
    n_pulls = zeros(K)
    rng = MersenneTwister(seed)
    p_star = collect(1 ./ (1:K))
    p_star[1] = 1/2
    p_star /= sum(p_star)

    Rejects(K, seed, rng, sum_rewards, n_pulls, p_star, 1)
end

function next_arm(self::Rejects)
    me = self.sum_rewards ./ self.n_pulls
    sidx = sortperm(me, rev=true)

    target_cnt = (self.t - 1)*self.p_star

    idx = self.n_pulls .== 0
    if (any(idx))
        i_pull = findfirst(idx)
    else
        v = target_cnt - self.n_pulls[sidx]
        maxval = maximum(v)
        i_pull = rand(self.rng, sidx[findall(v .== maxval)])
    end
    i_pull
end

mutable struct Uniform <: BanditAlg  
    const K::Int64
    const seed::Int64
    const rng::Random.MersenneTwister
    const sum_rewards::Vector{Float64} 
    const n_pulls::Vector{Int64} 
    t::Int64

    function Uniform(K, seed=123)
        sum_rewards = zeros(K)
        n_pulls = zeros(K)
        rng = MersenneTwister(seed)
        new(K, seed, rng, sum_rewards, n_pulls, 1)
    end
end

function next_arm(self::Uniform)
    idx = self.n_pulls .== 0
    if (any(idx))
        i_pull = findfirst(idx)
    else
        min_pull = minimum(self.n_pulls)
        i_pull = rand(self.rng, findall(self.n_pulls .== min_pull))
    end
    i_pull
end


################################################################################
# value estimators
################################################################################

abstract type ValueEstimator end

function estimate_value(self::ValueEstimator, alg::BanditAlg)
    estimate_value(self::ValueEstimator, alg.sum_rewards ./ alg.n_pulls, alg.n_pulls)
end

function estimate_value(self::ValueEstimator, data)
    estimate_value(self::ValueEstimator, meansqueeze(data, 2), size(data,2)*ones(Int, size(data,1)))
end

struct Average   <: ValueEstimator end
function estimate_value(self::Average, avgs::Vector{T}, n_pulls::Vector{Int}) where {T <: Real}
    p = n_pulls / sum(n_pulls)
    # println(p'avgs)
    # stop
    p'avgs
end

struct TopAverage   <: ValueEstimator end
function estimate_value(self::TopAverage, avgs::Vector{T}, n_pulls::Vector{Int}) where {T <: Real}
    K = length(avgs)
    N = n_pulls[1]
    mu_max = maximum(avgs)
    
    survived_arms_vals = []
    thres = sqrt(4*log(K)/N)
    for i in 1:K
        if mu_max - avgs[i] < 2*thres
            push!(survived_arms_vals, avgs[i])
        end
    end

    # println(K)
    # println(sum(n_pulls))
    # println(mu_max)
    # println(thres)
    # println(avgs)
    # println(survived_arms_vals)
    # println(mean(survived_arms_vals))
    # println(mean(survived_arms_vals, dims=1))
    
    # stop
    return mean(survived_arms_vals)
end


struct Max       <: ValueEstimator end
function estimate_value(self::Max, avgs::Vector{T}, n_pulls::Vector{Int}) where {T <: Real} 
    maximum(avgs)
end

struct Double       <: ValueEstimator 
    seed::Int
    rng::Random.MersenneTwister
    function Double(seed=97)
        rng = MersenneTwister(seed)
        new(seed, rng)
    end
end
function estimate_value(self::Double, data)
    n = size(data)[2]
    n1 = convert(Int, floor(n/2))
    n2 = n - n1
    avg1 = meansqueeze(data[:,1:n1], 2)
    avg2 = meansqueeze(data[:,n1+1:end], 2)

    # TODO need to break ties
    i1 = argmax(avg1)
    me1 = avg2[i1]
    i2 = argmax(avg2)
    me2 = avg1[i2]

    (me1 + me2) / 2
end


struct Weighted  <: ValueEstimator 
    σ²::Float64
    M::Int # number of mc samples
    seed::Int
    rng::Random.MersenneTwister
    function Weighted(σ², M, seed=93)
        rng = MersenneTwister(seed)
        new(σ², M, seed, rng)
    end
end
function estimate_value(self::Weighted, avgs::Vector{T}, n_pulls::Vector{Int}) where {T <: Real} 
    posterior_var = 1 ./ n_pulls
    K = length(avgs)

    #- simulate
    cnt = zeros(Int, K)
    for m = 1:self.M
        posterior_samples = avgs + sqrt(self.σ²) * randn(self.rng, K) ./ sqrt.(n_pulls)
        i_m = argmax(posterior_samples)
        cnt[i_m] += 1
    end
    p = cnt / sum(cnt)

    #- return
    p'avgs
end

struct NaiveOracle <: ValueEstimator 
    function NaiveOracle(mu)
        @assert(maximum(mu) == mu[1])
        new()
    end
end
function estimate_value(self::NaiveOracle, avgs::Vector{T}, n_pulls::Vector{Int}) where {T <: Real} 
    avgs[1]
end

struct Haver  <: ValueEstimator 
    σ²::Float64
    delta::Float64
    function Haver(σ²,delta=0.95)
        new(σ², delta)
    end
end

function estimate_value(self::Haver, avgs::Vector{T}, n_pulls::Vector{Int}) where {T <: Real} 
    K = length(avgs)
    emp_max = maximum(avgs)
    survived = findall(@. emp_max - avgs <= sqrt(2*self.σ² / n_pulls *log(K^2/self.delta)))
    if sum(n_pulls) >= 20*50-1
#        @infiltrate
        nothing
    end
    mean(avgs[survived])
end

struct WeightedMs <: ValueEstimator 
    σ²::Float64
    booster::Float64
    function WeightedMs(σ², booster=1.0)
        new(σ², booster)
    end
end
function estimate_value(self::WeightedMs, avgs::Vector{T}, n_pulls::Vector{Int}) where {T <: Real} 
    posterior_var = 1 ./ n_pulls

    i_max = argmax(avgs)
    hatDelta = avgs[i_max] .- avgs
    p = @. exp(- 1/(2*self.σ²) * n_pulls * hatDelta^2)
    p[i_max] = self.booster
    p = p/sum(p)

    p'avgs
end

struct WeightedMsGen <: ValueEstimator 
    σ²::Float64
    booster::Float64
    function WeightedMsGen(σ², booster=1.0)
        new(σ², booster)
    end
end
function estimate_value(self::WeightedMsGen, avgs::Vector{T}, n_pulls::Vector{Int}) where {T <: Real} 
    posterior_var = self.σ² ./ n_pulls
    hat_sigsq = posterior_var

    i_max = argmax(avgs)
    hatDelta = avgs[i_max] .- avgs

    #---------------------
    #
    w0 = @. 1/hat_sigsq      # this is the same as average estimator!!
    x = @. sqrt(hatDelta^2/(2*(hat_sigsq + hat_sigsq[i_max])))

    #- the original.
    #                w = @. exp(- 1/(2*hat_sigsq)* hatDelta^2)

    # w = @. 4^(x==0)*exp(- x^2)

    #                w = @. min(1.0,exp(-x^2)/x)

    w = @. 0.5/(1+x)*exp(-x^2)

    #                w = @. 1/(1+x^2)

    #- using w0
    #                w = w0

    #- sometimes this works well
    #                w = @. w0 * exp(- 1/(2*(hat_sigsq))* hatDelta^2)

    #                w = @. w0 * exp(- x^2)

    #                 w = @. w0 * min(1.0,exp(-x^2)/x)

    #                w = @. w0 * min(1.0,exp(-x^2)/x)  # seems to be the best.....
    #                w = @. w0 * min(1.0,exp(-x^2))    # this does not make much difference.
    #                w = @. w0 * min(0.05,exp(-x^2))    # this does not make much difference.

    #                w = @. w0 * min(1.0,1.0/x)*exp(-x^2)

    #                w = @. w0 * 1/(1+x^2)

    w[i_max] *= self.booster
    w /= sum(w)

    w'avgs
end










################################################################################
# min max tree
################################################################################

abstract type GameNode{T} end

"""
using Nothing for T if we do not want anything.
"""
struct MaxNode{T} <: GameNode{T}
#    value::Float64
    children::Vector{GameNode{T}}
    parent::Union{GameNode{T},Nothing}
    data::T

    MaxNode{T}(parent::Union{GameNode{T},Nothing}, data::T) where T = new(Vector{GameNode{T}}(), parent, data)
end

struct MinNode{T} <: GameNode{T}
#    value::Float64
    children::Vector{GameNode{T}}
    parent::Union{GameNode{T},Nothing}
    data::T

    MinNode{T}(parent::Union{GameNode{T},Nothing}, data::T) where T = new(Vector{GameNode{T}}(), parent, data)
end

function ismaxnode(self::GameNode{T}) where T 
    typeof(self) == MaxNode{T}
end

mutable struct MyValue
    value::Float64
    MyValue(value::Float64) = new(value)
    MyValue() = new(NaN)
end

isleaf(node::GameNode{T}) where T = length(node.children) == 0

################################################################################
#--------- define game tree
################################################################################

abstract type Reward end
struct KjGaussian  <: Reward end
struct KjBernoulli <: Reward end

struct GameTree{T,R<:Reward}
    root::MaxNode{T}
    σ²::Float64
    rngof::Dict{GameNode{T},MersenneTwister}
end

# function pull(self::GameTree{T}, leaf::GameNode{T}) where T
#     @assert isleaf(leaf)
#     leaf.data.value + sqrt(self.σ²)*randn(self.rngof[leaf])
# end

function pull(self::GameTree{T,KjGaussian}, leaf::GameNode{T}) where T
    @assert isleaf(leaf)
    leaf.data.value + sqrt(self.σ²)*randn(self.rngof[leaf])
end

function pull(self::GameTree{T,KjBernoulli}, leaf::GameNode{T}) where T
    @assert isleaf(leaf)
    convert(Float64, rand(self.rngof[leaf]) < leaf.data.value)
end

"""
root_type: MaxNode|MinNode
"""
function gen_toy_minmax_tree(depth, width, σ², T::Type, R::Type; seed=910) 
    rng = MersenneTwister(seed)
    rngof = Dict{GameNode{T},MersenneTwister}()
    root = gen_toy_minmax_tree(depth, width, MaxNode{T}, nothing, rngof, rng)

    GameTree{T,R}(root, σ², rngof)
end

function gen_toy_minmax_tree(depth, width, root_type::Type, parent, rngof::Dict{GameNode{T},MersenneTwister}, rng) where T
    if (depth == 0) # at the leaf node
        node = root_type(parent, T(rand(rng)))
        rngof[node] = MersenneTwister(randseed(rng))
        return node
    end

    if root_type == MaxNode{T}
        node = MaxNode{T}(parent, T())
        for i in 1:width
            push!(node.children, gen_toy_minmax_tree(depth - 1, width, MinNode, node, rngof, rng))
        end
        value = maximum([c.data.value for c in node.children])
        node.data.value = value
        return node
    elseif root_type == MinNode{T}
        node = MinNode{T}(parent, T())
        for i in 1:width
            push!(node.children, gen_toy_minmax_tree(depth - 1, width, MaxNode, node, rngof, rng))
        end
        value = minimum([c.data.value for c in node.children])
        node.data.value = value
        return node
    else
        @error "invalid value for root_type"
    end
end

function gen_toy_maxmax_tree(depth, width, σ², T::Type, R::Type; seed=910)
    rng = MersenneTwister(seed)
    rngof = Dict{GameNode{T},MersenneTwister}()
    root = gen_toy_maxmax_tree(depth, width, nothing, rngof, rng)

    GameTree{T,R}(root, σ², rngof)
end

function gen_toy_maxmax_tree(depth, width, parent, rngof::Dict{GameNode{T},MersenneTwister}, rng) where T
    if (depth == 0) # at the leaf node
        node = MaxNode{T}(parent, T(rand(rng)))
        rngof[node] = MersenneTwister(randseed(rng))
        return node
    end

    node = MaxNode{T}(parent, T())
    for i in 1:width
        push!(node.children, gen_toy_maxmax_tree(depth - 1, width, node, rngof, rng))
    end
    value = maximum([c.data.value for c in node.children])
    node.data.value = value
    return node
end



function print_node(depth::Int, root::GameNode{T}, b_lastchildren::Vector{Bool}) where T
    # ├──, └──, │  

    sb = ""
    if depth >= 1
        for i in 1:length(b_lastchildren)
            b = b_lastchildren[i]
            if (i != length(b_lastchildren))
                sb *= b ? "   " : "│  "
            else
                sb *= b ? "└──" : "├──" 
            end
        end
    end

    nodetype_str = ""
    if typeof(root) <: MaxNode
        nodetype_str = "∨" 
    else
        nodetype_str = "∧"
    end

    if isnan(root.data.value)
        println(sb * nodetype_str) 
    else
        println(sb * nodetype_str * (@sprintf ":%.3f" root.data.value))
    end

    for i in eachindex(root.children)
        c = root.children[i]
        blc = copy(b_lastchildren)
        append!(blc, i == length(root.children))
        print_node(depth+1, c, blc)
    end
end
print_node(tree::GameTree) = print_node(0, tree.root, zeros(Bool,0))


#----------------- MCTS -----------------------

#-----

abstract type MctsAlg end

mutable struct UctData
    value::Float64
    t::Int64 # this will be the number of arm pulls as well.
    sum_rewards::Float64
    hat_mu::Float64
    hat_sigsq::Float64

    UctData() = new(NaN, 0, NaN, NaN, Inf)
    UctData(value::Float64) = new(value, 0, NaN, NaN, Inf)
end


struct Uct <: MctsAlg 
    tree::GameTree{UctData,<:Reward}
    c::Float64
    σ²::Float64
    estimator::Symbol # :avg or :wems
    rng::Random.MersenneTwister

    function Uct(tree::GameTree{UctData,<:Reward}, c, σ²::Real, estimator; seed=928) 
#        node2alg = init_node2alg(tree.root, (K,reverse) -> Ucb(K, σ², reverse, randseed(rng), c=c))
        rng = MersenneTwister(seed) 
        new(tree, c, σ², estimator, rng)
    end
end


function next_leaf(self::MctsAlg)
    tree = self.tree
    node = tree.root
    while (~isleaf(node))
        i_next = next_arm(self, node)

        node = node.children[i_next]
    end
    node
end

global kj_warned
kj_warned = false
function next_arm(self::Uct, node::GameNode{T}) where T
    #- gather rewards
    hat_mu    = [c.data.hat_mu for c in node.children]
    hat_sigsq = [c.data.hat_sigsq for c in node.children]
    n_pulls = [c.data.t for c in node.children]

    i_next = findfirst(hat_mu .=== NaN) # FIXME break ties!!!
    if (i_next == nothing)
        if self.estimator == :wems_half
            width = self.c*sqrt.(2 .* self.σ² ./ n_pulls .* log(2 + node.data.t) )
        elseif self.estimator == :wems
            width = self.c*sqrt.(2 .* hat_sigsq .* log(max(exp(1), 1 + node.data.t) ))
#            width = self.c*sqrt.(1 .* self.σ² ./ n_pulls .* log(max(exp(1), 1 + node.data.t) ))
#            width = self.c*sqrt.(1 .* hat_sigsq .* sqrt.(1 + node.data.t) )
#            width = self.c*(4 .* hat_sigsq .* sqrt.(1 + node.data.t) ) # this one blocks :average from performing better after some time step.
        else
            width = self.c*sqrt.(2 .* hat_sigsq .* log(max(exp(1), 1 + node.data.t) ))
#            width = self.c*sqrt.(2 .* hat_sigsq .* sqrt.(1 + node.data.t) )
#            width = self.c*(4 .* hat_sigsq .* sqrt.(1 + node.data.t) ) # this one blocks :average from performing better after some time step.
            global kj_warned
            if (!kj_warned)
                println("WARNING: uct: using sqrt instead of log")
                kj_warned = true
            end
        end
        
        if ismaxnode(node)
            ucbs = hat_mu .+ width
            maxval = maximum(ucbs)
            i_next = rand(self.rng, findall(ucbs .== maxval))
        else
            lcbs = hat_mu .- width
            minval = minimum(lcbs)
            i_next = rand(self.rng, findall(lcbs .== minval))
#            i_next = argmin(lcbs)
        end
    end
    i_next
end

function update!(self::MctsAlg, leaf::GameNode{T}, reward::Float64) where T
    #- backup until reaching the root
    node = leaf
    tree = self.tree
    prev_node = nothing
    while (node != nothing)
        
        data = node.data
        if self.estimator == :average
            if data.t == 0
                data.sum_rewards = reward
            else
                data.sum_rewards += reward 
            end
            data.t += 1
            data.hat_mu = data.sum_rewards / (data.t)
            data.hat_sigsq = self.σ² / (data.t)
        elseif self.estimator == :wems_half
            if data.t == 0
                data.sum_rewards = reward
            else
                data.sum_rewards += reward 
            end
            
            data.t += 1
            if (isleaf(node))
                data.hat_mu = data.sum_rewards / data.t
                data.hat_sigsq = self.σ² / data.t
            else
                hat_mu = [c.data.hat_mu for c in node.children]
                hat_sigsq = [c.data.hat_sigsq for c in node.children]
                if (ismaxnode(node))
                    hDelta = maximum(hat_mu) .- hat_mu
                else
                    hDelta = hat_mu .- minimum(hat_mu)
                end
                w = @. exp(- 1/(2*hat_sigsq)* hDelta^2)
                w /= sum(w)
                data.hat_mu = w'hat_mu
                data.hat_sigsq = (w.^2)'hat_sigsq
            end
        
        elseif self.estimator == :wems
            if data.t == 0
                data.sum_rewards = reward
            else
                data.sum_rewards += reward 
            end
            
            data.t += 1
            if (isleaf(node))
                data.hat_mu = data.sum_rewards / data.t
                data.hat_sigsq = self.σ² / data.t
            else
                hat_mu    = [c.data.hat_mu    for c in node.children]
                ii = .~(isnan.(hat_mu))
                hat_mu = hat_mu[ii]
                hat_sigsq = [c.data.hat_sigsq for c in node.children]
                hat_sigsq = hat_sigsq[ii]

                i_star = nothing
                if (ismaxnode(node))
                    i_star = argmax(hat_mu)
                    hDelta = hat_mu[i_star] .- hat_mu
                else
                    i_star = argmin(hat_mu)
                    hDelta = hat_mu .- hat_mu[i_star]
                end
                w0 = @. 1/hat_sigsq      # this is the same as average estimator!!
                x = @. sqrt(hDelta^2/(2*(hat_sigsq + hat_sigsq[i_star])))

                #- the original.
#                w = @. exp(- 1/(2*hat_sigsq)* hDelta^2)

                w = @. 4^(x==0.0)*exp(- x^2)

#                w = @. min(1.0,exp(-x^2)/x)

#                w = @. min(1.0,1.0/x)*exp(-x^2)

#                w = @. 1/(1+x^2)

                #- using w0
#                w = w0
                
                #- sometimes this works well
#                w = @. w0 * exp(- 1/(2*(hat_sigsq))* hDelta^2)

#                w = @. w0 * exp(- x^2)
                
#                 w = @. w0 * min(1.0,exp(-x^2)/x)

#                w = @. w0 * min(1.0,exp(-x^2)/x)  # seems to be the best.....
#                w = @. w0 * min(1.0,exp(-x^2))    # this does not make much difference.
#                w = @. w0 * min(0.05,exp(-x^2))    # this does not make much difference.

#                w = @. w0 * min(1.0,1.0/x)*exp(-x^2)

#                w = @. w0 * 1/(1+x^2)

                w /= sum(w)
                data.hat_mu = w'hat_mu
                data.hat_sigsq = (w.^2)'hat_sigsq
                #@infiltrate
            end
        else
            @error "self.estimator is not valid"
        end

        prev_node = node
        node = node.parent
    end
end


struct Luct <: MctsAlg
    tree::GameTree{UctData,<:Reward}
    c::Float64
    σ²::Float64
    estimator::Symbol # :avg or :wems
    rng::Random.MersenneTwister

    function Luct(tree::GameTree{UctData,<:Reward}, c, σ²::Float64, estimator; seed=928) 
        rng = MersenneTwister(seed) 
        new(tree, c, σ², estimator, rng)
    end
end

function next_arm(self::Luct, node::GameNode{T}) where T
    #- gather rewards
    hat_mu    = [c.data.hat_mu for c in node.children]
    hat_sigsq = [c.data.hat_sigsq for c in node.children]

    i_next = findfirst(hat_mu .=== NaN) # FIXME break ties!!!
    if (i_next == nothing)
        n_pulls = [c.data.t for c in node.children]
        if self.estimator == :wems_half
            width = self.c*sqrt.(2 .* self.σ² ./ n_pulls .* log(2 + node.data.t) )
#         elseif self.estimator == :wems
#             width = self.c*sqrt.(2 .* hat_sigsq .* log(2 + node.data.t) )
        else
#            width = self.c*(1 .* hat_sigsq .* sqrt.(1 + node.data.t) ) # this one blocks :average from performing better after some time step.
            width = self.c*sqrt.(2 .* self.σ² ./ n_pulls .* log(2 + node.data.t) )

        end
        
        b_max = ismaxnode(node)

        ucb = hat_mu .+ width
        lcb = hat_mu .- width
        
        if (b_max)
            i_best = argmax_breaktie(self.rng, hat_mu)
            idx = trues(length(hat_mu))
            idx[i_best] = false
            bot = findall(idx)

            i_bestbot = bot[argmax_breaktie(self.rng, ucb[bot])]

            candidates = [i_best, i_bestbot]
        else
            i_best = argmin_breaktie(self.rng, hat_mu)
            idx = trues(length(hat_mu))
            idx[i_best] = false
            top = findall(idx)

            i_worsttop = top[argmin_breaktie(self.rng, lcb[top])]

            candidates = [i_best, i_worsttop]
        end
#        i_next = candidates[argmax(hat_sigsq[candidates])]
        i_next = rand(self.rng, candidates)
    end
    i_next
end




struct Ut <: MctsAlg
    tree::GameTree{UctData,<:Reward}
    σ²::Float64
    estimator::Symbol # :avg or :wems
    rng::Random.MersenneTwister

    function Luct(tree::GameTree{UctData,<:Reward}, σ²::Float64, estimator; seed=928) 
        rng = MersenneTwister(seed) 
        new(tree, σ², estimator, rng)
    end
end

function next_arm(self::Ut, node::GameNode{T}) where T
    #- gather rewards
    n_pulls = [c.data.t for c in node.children]
    i_next = argmin_breaktie(self.rng, n_pulls)
    i_next
end

struct Srt <: MctsAlg
    tree::GameTree{UctData,<:Reward}
#    c::Float64
    σ²::Float64
    c_srt::Float64
    estimator::Symbol # :avg or :wems
    rng::Random.MersenneTwister

    function Srt(tree::GameTree{UctData,<:Reward}, σ²::Real; c_srt=1.0, seed=928) 
#        node2alg = init_node2alg(tree.root, (K,reverse) -> Ucb(K, σ², reverse, randseed(rng), c=c))
        rng = MersenneTwister(seed) 
        new(tree, σ², c_srt, :wems, rng)
    end
end

function next_arm(self::Srt, node::GameNode{T}) where T
    #- gather rewards
    hat_mu    = [c.data.hat_mu for c in node.children]
    n_pulls   = [c.data.t      for c in node.children] 
    
    i_next = findfirst(hat_mu .=== NaN) # FIXME break ties!!!
    if (i_next == nothing)
        rev = ismaxnode(node)
        sidx = sortperm(hat_mu, rev=rev)

        p_star = collect(1 ./ (1:length(hat_mu)))
        p_star[1] = self.c_srt*p_star[2]
#        p_star[1] = p_star[2]
        p_star /= sum(p_star)

        target_cnt = (node.data.t + 1)*p_star

        v = target_cnt .- n_pulls[sidx]
        maxval = maximum(v)
        i_next = rand(self.rng, sidx[findall(v .== maxval)])
    else
        nothing
#        @infiltrate
    end
    i_next
end

using Printf

function print_children_data(node)
    for i in eachindex(node.children)
        c = node.children[i]
        c.data
        @printf "i=%4d: %s\n" i repr(c.data)
    end
end















