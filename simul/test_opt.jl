include("utils.jl")
function create_Alpha(K1, K2)
    res = Float64[]
    while length(res) < K1*K2
        r = rand(Normal(1.0/(K1*K2), 1.0/(K1*K2)^2))
        # r = rand(1)[1]
        if r > 0
            res = vcat(res, r)
        else
			continue
		end
    end
    Res = convert(Matrix{Float64}, transpose(reshape(res, (K2, K1))))
    return res, Res
end


function create_Theta(vec, N, K1, K2)
    res = convert(Matrix{Float64},transpose(rand(Distributions.Dirichlet(vec),N)))
    Res = [convert(Matrix{Float64},transpose(reshape(res[i,:], (K2, K1)))) for i in 1:N]
    Res1 = [sum(Res[i], dims=2)[:,1] for i in 1:N]
    Res2 = [sum(Res[i], dims=1)[1,:] for i in 1:N]
    return res, Res, Res1, Res2
end

function create_phis!(phi, Thetax, N)
    for i in 1:N
		wlen, K = size(phi[i])
        for w in 1:wlen
            s = sum(rand(Distributions.Multinomial(1,Thetax[i]), 100), dims=2)[:,1] .*1.0
            # s ./= sum(s)
            phi[i][w,:] = normalize(s,1)
        end
    end
end


K1 = 20
K2 = 30
N = 500
alpha_vec, Alpha = create_Alpha(K1, K2)
Plots.heatmap(Alpha)
Theta_vec, Theta, Theta1, Theta2 = create_Theta(alpha_vec, N, K1, K2)

fixed_len1 = 100 .*ones(Int64, N)
fixed_len2 = 100 .*ones(Int64, N)
# fixed_len1 = Int64.(ceil.(rand(Exponential(.1), N).*1000))
# fixed_len2 = Int64.(ceil.(rand(Exponential(.08), N).*1000))
phi1 = [zeros(Float64, (fixed_len1[i],K1)) for i in 1:N]
phi2 = [zeros(Float64, (fixed_len2[i],K2)) for i in 1:N]
create_phis!(phi1, Theta1, N)
create_phis!(phi2, Theta2, N)
# [sum(x) for x in phi1]

γ = [zeros(Float64, K1, K2) for i in 1:N]
for i in 1:N
	γ[i] = deepcopy(Alpha)
end

function compute_ℒ_γ_atom(K1_, K2_, k1_, k2_, Alpha_, γ_, phi1_, phi2_)##gamma  and phis are i-indexed
    ℒ_γ = 0.0
    ℒ_γ += (Alpha_[k1_, k2_]-γ_[k1_, k2_])*(digamma_(γ_[k1_, k2_]))
    for l1 in 1:K1_
        for l2 in 1:K2_
            ℒ_γ -= (Alpha_[l1, l2]-γ_[l1, l2])*digamma_(sum(γ_))
        end
    end
    ℒ_γ -= lgamma_(sum(γ_))
    ℒ_γ += (lgamma_(γ_[k1_, k2_]))

    for w in 1:size(phi1_,1)
        ℒ_γ += phi1_[w,k1_] * (digamma_(sum(γ_[k1_,:])))
        for l1 in 1:K1_
            ℒ_γ -= phi1_[w,l1] * digamma_(sum(γ_))
        end
    end
    for w in 1:size(phi2_,1)
        ℒ_γ += phi2_[w,k2_]*(digamma_(sum(γ_[:,k2_])))
        for l2 in 1:K2_
            ℒ_γ -= phi2_[w,l2] * digamma_(sum(γ_))
        end
    end
    return ℒ_γ
end

function compute_∇ℒ_γ_atom(K1_, K2_, k1_, k2_,Alpha_,γ_,phi1_, phi2_) # indexed at i
    rest = 0.0
    rest += (Alpha_[k1_, k2_]-γ_[k1_, k2_])*(trigamma_(γ_[k1_, k2_]))
    for w in 1:size(phi1_,1)
        rest += phi1_[w,k1_]*trigamma_(sum(γ_[k1_,:]))
    end
    for w in 1:size(phi2_,1)
        rest += phi2_[w,k2_]*trigamma_(sum(γ_[:,k2_]))
    end

    special_term = 0.0
    for l1 in 1:K1_
        for l2 in 1:K2_
            special_term += (Alpha_[l1, l2]-γ_[l1, l2])
        end
    end
    special_term += (size(phi1_,1)+size(phi2_,1))*1.0

	special_term *= -trigamma_(sum(γ_))
    ∇_γ = rest + special_term
    return ∇_γ , rest , special_term
end


function optimize_γ_i(K1_, K2_,old_, Alpha_,phi1_, phi2_, ITER)
    # γ_running = zeros(Float64, (K1_, K2_))
	γ_running = deepcopy(old_)
	for k1 in 1:K1_
		for k2 in 1:K2_
			sum1 = sum([phi1_[w,k1] for w in 1:size(phi1_,1)])
			sum2 = sum([phi2_[w,k2] for w in 1:size(phi2_,1)])
			for iter in 1:ITER
				γ_running[k1,k2] = Alpha_[k1, k2] + sum1 * (γ_running[k1, k2]/sum(γ_running[k1,:]))
				γ_running[k1,k2] = Alpha_[k1, k2] + sum2 * (γ_running[k1, k2]/sum(γ_running[:,k2]))
			end
		end
	end
	return γ_running
end

prior_grad_results = zeros(Float64, N)
for i in 1:N
	prior_grad_results[i] = mean([compute_∇ℒ_γ_atom(K1, K2, k1, k2,Alpha,γ[i],phi1[i], phi2[i])[1]
	 for k1 in 1:K1 for k2 in 1:K2])
end
Plots.histogram(prior_grad_results)
grad_results = zeros(Float64, N)
for i in 1:N
	if i % 10 == 0
		println(i)
	end
	γ[i] = optimize_γ_i(K1, K2,γ[i], Alpha,phi1[i], phi2[i], 20)
	grad_results[i] = mean([compute_∇ℒ_γ_atom(K1, K2, k1, k2,Alpha,γ[i],phi1[i], phi2[i])[1]
	 for k1 in 1:K1 for k2 in 1:K2])
end
mean(grad_results)
median(grad_results)

est_theta1 = [[sum(γ[i][k1,:])/sum(γ[i]) for k1 in 1:K1] for i in 1:N]
est_theta2 = [[sum(γ[i][:,k2])/sum(γ[i]) for k2 in 1:K2] for i in 1:N]
mean([mean(abs.(est_theta1[i] .- Theta1[1])) for i in 1:N])
mean([mean(abs.(est_theta2[i] .- Theta2[1])) for i in 1:N])

Plots.histogram(grad_results)
describe(grad_results)

Plots.heatmap(Alpha)
compute_∇ℒ_γ_atom(K1, K2, 3, 20,Alpha,γ[10],phi1[10], phi2[10])
compute_ℒ_γ_atom(K1, K2, 1, 2, Alpha, γ[1], phi1[1], phi2[1])
