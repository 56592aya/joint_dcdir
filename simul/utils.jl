using Random
using Distributions
using DataStructures
using StatsBase
using StatsPlots
using Plots
using LinearAlgebra
using SpecialFunctions

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

function digamma_(x::Float64)
	p=zero(Float64)
  	x=x+6.0
  	p=1.0/(x*x)
  	p=(((0.004166666666667*p-0.003968253986254)*p+0.008333333333333)*p-0.083333333333333)*p
  	p=p+log(x)-0.5/x-1.0/(x-1.0)-1.0/(x-2.0)-1.0/(x-3.0)-1.0/(x-4.0)-1.0/(x-5.0)-1.0/(x-6.0)
  	p
end

function lgamma_(x::Float64)
	z=1.0/(x*x)
 	x=x+6.0
	z=(((-0.000595238095238*z+0.000793650793651)*z-0.002777777777778)*z+0.083333333333333)/x
    z=(x-0.5)*log(x)-x+0.918938533204673+z-log(x-1)-log(x-2)-log(x-3)-log(x-4)-log(x-5)-log(x-6);
  z
end

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
    log(r) + alpha
end
function logsumexp(X::Matrix{Float64})
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
    log(r) + alpha
end
function softmax(X::Vector{Float64})
    lse = logsumexp(X)
    return  @.(exp(X-lse))
end
function softmax(X::Matrix{Float64})
    lse = logsumexp(X)
    return  @.(exp(X-lse))
end

function softmax2(MEM::Vector{Float64},X::Vector{Float64})
    lse = logsumexp(X)
    @.(MEM = (exp(X - lse)))
end

# sum(softmax([1.0 3.0; 2.0 4.0]))

println("utils.jl loaded")
