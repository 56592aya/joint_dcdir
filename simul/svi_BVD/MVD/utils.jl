using BenchmarkTools
using Random
using Distributions
using LinearAlgebra

VectorList{T} = Vector{Vector{T}}
MatrixList{T} = Vector{Matrix{T}}
mutable struct Document
	terms::Vector{Int64}
	counts::Vector{Int64}
	len::Int64
end
mutable struct Corpus
	docs::Vector{Document}
	N::Int64
	V::Int64
end


function digamma_(x::Float64)
	p=zero(Float64)
  	x=x+6.0
  	p=1.0/(x*x)
  	p=(((0.004166666666667*p-0.003968253986254)*p+0.008333333333333)*p-0.083333333333333)*p
  	p=p+log(x)-0.5/x-1.0/(x-1.0)-1.0/(x-2.0)-1.0/(x-3.0)-1.0/(x-4.0)-1.0/(x-5.0)-1.0/(x-6.0)
  	p
end

function Elog(Mat::Matrix{Float64})
    digamma_.(Mat) .- digamma_(sum(Mat))
end
function Elog(Vec::Vector{Float64})
    digamma_.(Vec) .- digamma_(sum(Vec))
end


function logsumexp(X::Vector{Float64})

    alpha = -Inf::Float64;
	r = 0.0;
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
    alpha = -Inf::Float64;
	r = 0.0;
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
    alpha = -Inf::Float64;
	r = 0.0;
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

function softmax(MEM::Matrix{Float64},X::Matrix{Float64})
    lse = logsumexp(X)
    @.(MEM = (exp(X - lse)))
end
function softmax(X::Matrix{Float64})
    return exp.(X .- logsumexp(X))
end


function sort_by_argmax!(X::Matrix{Float64})

	n_row=size(X,1)
	n_col = size(X,2)
	ind_max=zeros(Int64, n_row)
	permuted_index = zeros(Int64, n_row)
	for a in 1:n_row
    	ind_max[a] = findmax(view(X,a,1:n_col))[2]
	end
	X_tmp = similar(X)
	count_ = 1
	for j in 1:maximum(ind_max)
  		for i in 1:n_row
    		if ind_max[i] == j
	      		for k in 1:n_col
	        		X_tmp[count_, k] = X[i,k]
	      		end
				permuted_index[count_]=i
      			count_ += 1
    		end
  		end
	end
	# This way of assignment is important in arrays, el by el
	X[:]=X_tmp[:]
	X, permuted_index
end
