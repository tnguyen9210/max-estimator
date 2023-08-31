
"""
goal: let's try out the segal'12 estimator.
- consider the frequentist setup.
"""

using Distributions
using Random

#Random.seed!(19)

n = 10_000
var1 = 1.0
var2 = 1.0
me1 = 0.0
me2 = -0.0
X1 = me1 .+ randn(n).*sqrt(var1)
X2 = me2 .+ randn(n).*sqrt(var2)
X = hcat(X1, X2)
#X = randn(rng, n, 2) 


sig_m = sqrt(var1 + var2)
alpha = (X[:,1] - X[:,2])/sig_m
mynormal = Normal()
Y1 = @. X[:,1]*cdf(mynormal, alpha) + X[:,2]*cdf(mynormal, -alpha) + sig_m * pdf(mynormal, alpha)

maxidx = argmax.(eachrow(X))
Delta = maximum.(eachrow(X)) .- X
function w_wems(v)
    Delta = maximum(v) .- v
#    w = @. exp(-1/(2*(var1 + var2))*Delta.^2)
    w = @. 4^(Delta==0)*exp(-1/(2*(var1 + var2))*Delta.^2)
    w /= sum(w)
end
function w_we(v)
    A1 = randn(100).*sqrt(var1) .+ v[1]
    A2 = randn(100).*sqrt(var2) .+ v[2]
    A = hcat(A1, A2)
    aa = argmax.(eachrow(A))
    w = mean(aa .== 1)
    [w,1-w]
end
W2 = w_wems.(eachrow(X))
W2 = mapreduce(permutedims, vcat, W2)
W3 = w_we.(eachrow(X)) 
W3 = mapreduce(permutedims, vcat, W3)

sumsqueeze(X,d) = dropdims(sum(X,dims=d), dims=d)

Y2 = sumsqueeze(X .* W2, 2)
myvar2 = sumsqueeze(W2.^2 .* [var1, var2]', 2)
Y3 = sumsqueeze(X .* W3, 2)
myvar3 = sumsqueeze(W3.^2 .* [var1, var2]', 2)

@show (mean(Y1.^2), mean(Y1), var(Y1))
@show (mean(Y2.^2), mean(Y2), var(Y2))
@show (mean(Y3.^2), mean(Y3), var(Y3))

@show (mean(myvar2), var(myvar2))
@show (mean(myvar3), var(myvar3))

