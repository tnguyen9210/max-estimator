function problem_factory(
    problem_name::String, σ²::Real, seed::UInt32=987;
    K=-1, alpha=1.0, Delta=0.1, splits=0.5)
    if  problem_name == "linear"
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
    elseif problem_name == "alpha_frac_single_gap"
        alphaK = Int(floor(0.3*K))
        mu = zeros(K)
        mu[alphaK+1:K] .= -1.0/K
        problem = Bandit(mu, σ², seed)
    elseif problem_name == "all_best_last_odd"
        mu = zeros(K)
        mu[K] = -1.0/K
        problem = Bandit(mu, σ², seed)
    elseif problem_name == "linear_K"
        mu = collect(range(start=0, length=K, step=-1.0/K))
        problem = Bandit(mu, σ², seed)
    elseif problem_name == "linear_K_splits"
        if isnothing(Delta) || isnan(Delta)
            Delta = 1.0/K
        end
        mu = collect(range(start=0, length=K, step=-Delta))
        if typeof(splits) == Float64
            split_pos = Int(floor(splits*K))
            mu[split_pos+1:end] .-= 5.0
        elseif typeof(splits) == Int64
            mu[splits+1:end] .-= 5.0
        elseif typeof(splits) == Vector{Int64}
            for i in 1:length(splits)
                mu[splits[i]+1:end] .-= 5.0
            end
        end
        # println(Delta)
        # println(mu)
        # stop
        problem = Bandit(mu, σ², seed)
    elseif problem_name == "randomized"
        tmp = rand(MersenneTwister(111), K)
        mu = (tmp .* 2.0) .- 2.0
        mu[1] = 0.0
        problem = Bandit(mu, σ², seed)
    end
    problem
end

function estimator_factory(name, σ², mu=Float64[], seed=123)
    if (name == "avg")
        est = Average()
    elseif name == "topavg"
        est = TopAverage()
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
        est = Haver(σ²,0.05) # 0.01, 0.05, 0.10
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
    else
        @error "value error"
    end
    algo
end
