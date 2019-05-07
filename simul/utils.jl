using Distributions
using DataStructures
using StatsBase
using StatsPlots
using Plots
using LinearAlgebra
#single value
function trigamma_(x::Float64)
    x += 6;
    p = 1.0 /(x*x)
    p=(((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)
         *p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p
    for i in 1:6
        x -= 1
        p= 1.0/(x*x)+p
	end
    p
end


# vs = (collect(LinRange(0.0, 100.0, 201)))[2:end]
# valst = [trigamma_(v) for v in vs]
# valsd = [digamma_(v) for v in vs]
# valsl = [lgamma_(v) for v in vs]
# Plots.plot(vs, valst)
# Plots.plot(vs, valsd)
# Plots.plot(vs, valsl)

function digamma_(x::Float64)
	p=zero(Float64)
  	x=x+6.0
  	p=1.0/(x*x)
  	p=(((0.004166666666667*p-0.003968253986254)*p+0.008333333333333)*p-0.083333333333333)*p
  	p=p+log(x)-0.5/x-1.0/(x-1.0)-1.0/(x-2.0)-1.0/(x-3.0)-1.0/(x-4.0)-1.0/(x-5.0)-1.0/(x-6.0)
  	p
end
#single_value
function lgamma_(x::Float64)
	z=1.0/(x*x)
 	x=x+6.0
  	z=(((-0.000595238095238*z+0.000793650793651)*z-0.002777777777778)*z+0.083333333333333)/x
  	z=(x-0.5)*log(x)-x+0.918938533204673+z-log(x-1.0)-log(x-2.0)-log(x-3.0)-log(x-4.0)-log(x-5.0)-log(x-6.0)
  	z
end

# give me a K1*K2 matrix and I give you row sums(topic wise) or col sums(community wise)
function get_1dim_dist(M::Matrix{Float64}, cnr::Int64)
	if (cnr == 1)
		return sum(M, dims=2)[:,1]
	elseif (cnr == 2)
		return sum(M, dims=1)[1,:]
	else
		println("asking for a bad thing and I can't do it!")
	end
end
function get_1dim_dist(M::Matrix{Float64}, cstr::String)
	if (lowercase(cstr) in ["t","topic"])
		return sum(M, dims=2)[:,1]
	elseif (lowercase(cstr) in ["c", "comm", "com", "community"])
		return sum(M, dims=1)[1,:]
	else
		println("asking for a bad thing and I can't do it!")
	end
end

function Elog(par1, par2)
	return (digamma_(par1) - digamma_(par2))
end

function logsumexp(X::Vector{Float64})

    alpha = -Inf;r = 0.0;
    @inbounds for x in X
        if x <= alpha
            r += exp(x - alpha)
        else
            r *= exp(alpha - x)
            r += 1.0
            alpha = x
        end
    end
    # log1p(r-1.0) + alpha
    log(r) + alpha
end
function logsumexp(X::Float64, Y::Float64)
    alpha = -Inf;r = 0.0;
    if X <= alpha
        r += exp(X - alpha)
    else
        r *= exp(alpha - X)
        r += 1.0
        alpha = X
    end
    if Y <= alpha
        r += exp(Y - alpha)
    else
        r *= exp(alpha - Y)
        r += 1.0
        alpha = Y
    end
    # log1p(r-1.0) + alpha
    log(r) + alpha
end
function softmax(X::Vector{Float64})
    lse = logsumexp(X)
    return  @.(exp(X-lse))
end
###Surprisingly the log(a/b) is more stable that log(a) - log(b)
function softmax2(MEM::Vector{Float64},X::Vector{Float64})
    lse = logsumexp(X)
    @.(MEM = (exp(X - lse)))
end
function alternate_gamma(k1::Int64, k2::Int64, sum1::Float64, sum2::Float64, ALPHA::Float64, MAXINNER::Int64)
	##Also find a way to check gradient decreasing
	γ_running = ALPHA
	for i in 1:MAXINNER
		γ_running[k1,k2] = ALPHA[k1, k2] + sum1*(γ_running[k1, k2]/sum(γ_running[k1,:]))
		γ_running[k1,k2] = ALPHA[k1, k2] + sum2*(γ_running[k1, k2]/sum(γ_running[:,k2]))
	end
return (γ_running)
end

println("utils.jl loaded")
