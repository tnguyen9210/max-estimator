# module KjUtils

# export meshgrid, normalizerows, normalizecols, binarysearch, sumsqueeze, meansqueeze, randseed, argmax_breaktie, argmin_breaktie

using Statistics

sumsqueeze(A, dims) = dropdims(sum(A, dims=dims); dims=dims)
meansqueeze(A, dims) = dropdims(mean(A, dims=dims); dims=dims)
stdsqueeze(A, dims) = dropdims(std(A, dims=dims); dims=dims)

function meshgrid(x, y)
   X = [x for _ in y, x in x]
   Y = [y for y in y, _ in x]
   X, Y
end

function normalizerows(A)
  m, n = size(A)
  norms = norm.(eachrow(A))
  A ./ reshape(norms, (m,1))
end

function normalizecols(A)
  m, n = size(A)
  norms = norm.(eachcol(A))
  A ./ reshape(norms, (1,n))
end

function binarysearch(fn, lb::Real, ub::Real; tol=1e-5)
  """
    lb and ub must satisfy fn(lb) < 0 && fn(ub) > 0 
  """
  fnlb = fn(lb)
  fnub = fn(ub)
  @assert fnlb * fnub != 0.0
  @assert ( fnlb <= 0 && fnub >= 0 ) || ( fnlb >= 0 && fnub <= 0)
  sign_fnlb = 1.0
  if (fnlb <= 0 && fnub >= 0)
    sign_fnlb = -1.0
  end
  max_iter = 1000
  i_iter = 0
  while ub-lb >= tol && i_iter <= max_iter
    mid = (lb+ub)/2
    val = fn(mid)
    if (val*sign_fnlb > 0.0) # when the sign of val is equal to that of fn(lb)
      lb = mid
    else
      ub = mid
    end
    i_iter += 1
  end
  if (i_iter >= max_iter)
    @warn "max_iter has reached"
  end
  [lb,ub]
end


function kl(p::Real,q::Real)
    @assert (0.0 <= p <= 1.0) && (0.0 <= q <= 1.0)
    if (p == 0.0)
        p = 0.0
        val = -log(1-q)
    elseif (p == 1.0)
        p = 1.0
        val = -log(q)
    else
        val = p*(log(p) - log(q)) + (1-p)*(log(1-p) - log(1-q))
    end
    val
end

function randseed(rng)
    rand(rng, UInt32)
end

function argmax_breaktie(rng, v)
    maxval = maximum(v)
    rand(rng, findall(v .== maxval))
end
function argmin_breaktie(rng, v)
    minval = minimum(v)
    rand(rng, findall(v .== minval))
end

"""
   vectorization.
"""
function vec_triu_loop(M::AbstractMatrix{T}) where T
    m, n = size(M)
    m == n || throw(error("not square"))
    l = n*(n+1) ÷ 2
    v = Vector{T}(l)
    k = 0
    @inbounds for i in 1:n
        for j in 1:i
            v[k + j] = M[j, i]
        end
        k += i
    end
    v
end


# end

