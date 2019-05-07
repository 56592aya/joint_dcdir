#Let's create the data after feeding the ground truth
ndocs=1000
nterms1 = 2000
nterms2 = 3500
K1=2
K2=3
alpha = .16
beta1 = 0.01
beta2 = 0.001
doclens1 = repeat([150], ndocs)
doclens2 = repeat([200], ndocs)

corpus1, corpus2, Theta,
B1, B2, Terms1, Terms2 =
            simulate_Bimodal(ndocs, nterms1,nterms2,
            K1, K2, alpha, beta1,beta2,
            doclens1, doclens2)
#Preliminary work for the inference
# I am NOT trying SVI now, so I'll scan through every thing for the time being

# I need to use these digammas, so better to use good ones, so include the utils.jl
include("utils.jl")
MAXITER = 1000
MAXINNER = 20
# We must have access to the ndocs, corpus1, corpus2, nterms1, nterms2, doclens1, doclens2
# let's assume we have the right number of the K1, and K2
# Creating containers for the variational parameters

γ = [zeros(Float64, (K1, K2)) for i in 1:ndocs]
# we have some thing like ϕ(iwk),
# do we need w dimension to be of length doclens[i]? or V
ϕ1 = [zeros(Float64, (length(Terms1),K1)) for i in 1:ndocs]
ϕ2 = [zeros(Float64, (length(Terms2),K2)) for i in 1:ndocs]
b1 = zeros(Float64, (K1, length(Terms1)))
b2 = zeros(Float64, (K2, length(Terms2)))
ElogTheta1 = zeros(Float64, (ndocs, K1))
ElogTheta2 = zeros(Float64, (ndocs, K2))
ElogB1 = zeros(Float64, (K1, nterms1))
ElogB2 = zeros(Float64, (K2, nterms2))
for iter in 1:MAXITER
    init_params()
    # 1) update phis first
    for i in 1:ndocs
        for w in 1:doclens1[i] #yes but what am I iterating over
            γ1 = get_1dim_dist(γ[i], "t")
            γ_all = sum(γ[i])
            res = zeros(Float64, K1)
            for k in 1:K1
                ElogTheta1[i,k] = Elog(γ1[k], γ_all)
                ElogB1[i,k] = Elog(b1[k,w], sum(b1[k,:])) ##the w here is not correct
                res[k] = ElogTheta1[i,k] + ElogB1[i,k]
            end
            # update phi1
            ϕ1[i][w,:] = softmax(res)
        end
        for w in 1:doclens2[i]
            γ2 = get_1dim_dist(γ[i], "c")
            γ_all = sum(γ[i])
            res = zeros(Float64, K2)
            for k in 1:K2
                ElogTheta2[i,k] = Elog(γ2[k], γ_all)
                ElogB2[i,k] = Elog(b2[k,w], sum(b2[k,:])) ##the w here is not correct
                res[k] = ElogTheta2[i,k] + ElogB2[i,k]
            end
            # update phi1
            for k in 1:K2
                ϕ2[i][w,k] = exp.(res[k] - logsumexp(res))
            end
        end
    end

    # 2) update the gammas
    for i in 1:ndocs
        #possibly numerically
        ALPHA = alpha*ones(Float64, (K1, K2))
        for k1 in 1:K1
            sum1 = (sum([ϕ1[i][w,k1]for w in 1:length(Terms1)]))
            for k2 in 1:K2
                sum2 = (sum([ϕ2[i][w,k2]for w in 1:length(Terms2)]))
                γ_[i][k1,k2] = alternate_gamma(k1, k2, sum1, sum2, ALPHA, MAXINNER)
            end
        end
    end

    # 3)update the b's
    for k in 1:K1
        for w in 1:length(Terms1)
            b1[k,w] = beta[k] +  sum([ϕ1[i][w,k] for i in 1:ndocs])
        end
    end

    for k in 1:K2
        for w in 1:length(Terms2)
            b2[k,w] = beta[k] +  sum([ϕ2[i][w,k] for i in 1:ndocs])
        end
    end

    # do some cleaning and estimations and then some evaluation

    # estimate Theta's
    # estimate B's
    # What needs to be modified?(ElogB's and ElogThetas?)
    # Eval model



end
