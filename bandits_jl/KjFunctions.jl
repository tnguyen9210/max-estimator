

function problem_factory(
    problem_name::String, σ²::Real, seed::UInt32=987;
    K=-1, num_optimals=-1, degree=1.0, Delta=0.1, gap_splits=0)
    if  problem_name == "all_best"
        mu = zeros(K)
        problem = Bandit(mu, σ², seed)
    elseif problem_name == "multi_gap_linear"
        # format gap_splits
        if typeof(gap_splits) == Float64
            gap_splits = [Int(floor(gap_splits*K))]
        elseif typeof(gap_splits) == Int64
            gap_splits = [gap_splits]
        elseif typeof(gap_splits) == Vector{Float64}
            gap_splits = [Int(floor(frac*K)) for frac in gap_splits]
        end

        # compute mu for each gap
        if gap_splits[1] == 0
            popfirst!(gap_splits)
        end
        if gap_splits[end] == K-1 || gap_splits[end] == K
            pop!(gap_splits)
        end
        push!(gap_splits, K-1)
        num_splits = length(gap_splits)
        gap_mu = -collect((0:num_splits) ./ (num_splits-1))
        
        mu = zeros(K)
        for i in 2:num_splits
            mu[gap_splits[i-1]+1:gap_splits[i]+1] .= gap_mu[i]
        end
        # pop!(gap_splits)

        # println(gap_splits)
        # println(gap_mu)
        # println(mu)
        # println(length(mu))
        # stop
        problem = Bandit(mu, σ², seed)
    elseif problem_name == "K_gap_poly"
        mu = -(((0:K-1) ./ (K-1)) .^ degree)
        problem = Bandit(mu, σ², seed)
    elseif problem_name == "multi_best_poly"
        mu = zeros(K)
        num_suboptimals = K - num_optimals + 1
        mu[num_optimals+1:end] .= -(((1:num_suboptimals-1) ./ (num_suboptimals-1)) .^ degree)
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
    elseif name == "double"
        est = Double(seed)
    elseif name == "weighted_100"
        est = Weighted(σ², 100, seed)
    elseif name == "weightedms"
        est = WeightedMs(σ²)
    elseif name == "weightedms-gen"
        est = WeightedMsGen(σ²,3)
    elseif name == "weightedms_4"
        est = WeightedMs(σ²,4)
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
