using Distributions
using DataStructures
using StatsBase
using StatsPlots
using Plots

function simulate_Bimodal(ndocs, nterms1,nterms2, K1, K2, alpha, beta1,beta2, doclens1, doclens2)



    Alpha = repeat([alpha], K1*K2)
    Beta1 = repeat([beta1], nterms1)
    Beta2 = repeat([beta2], nterms2)


    #creating labels
    Terms1 = ["Term1_$i" for i in 1:nterms1]
    Terms2 = ["Term2_$i" for i in 1:nterms2]
    Topics1 = ["Topic1_$k" for k in 1:K1]
    Topics2 = ["Topic2_$k" for k in 1:K2]
    Documents1 = ["Doc$d" for d in 1:ndocs]
    Documents2 = ["Doc$d" for d in 1:ndocs]

    Theta_vec = convert(Matrix{Float64},transpose(rand(Distributions.Dirichlet(Alpha),ndocs)))
    Theta = [convert(Matrix{Float64},transpose(reshape(Theta_vec[i,:], (K2, K1)))) for i in 1:ndocs]
    Theta1 = [sum(Theta[i], dims=2)[:,1] for i in 1:ndocs]
    Theta2 = [sum(Theta[i], dims=1)[1,:] for i in 1:ndocs]

    B1 = convert(Matrix{Float64},transpose(rand(Distributions.Dirichlet(Beta1), K1)))
    B2 = convert(Matrix{Float64},transpose(rand(Distributions.Dirichlet(Beta2), K2)))

    function create_single_doc(doclen, topic_dist, term_topic_dist, Terms)
        doc = String[]
        for i in 1:doclen
            topic = argmax(rand(Distributions.Multinomial(1,topic_dist)))
            term =  argmax(rand(Distributions.Multinomial(1,term_topic_dist[topic,:])))
            doc = vcat(doc, Terms[term])
        end
        return doc
    end
    corpus1 = [[] for i in 1:ndocs]
    corpus2 = [[] for i in 1:ndocs]
    for i in 1:ndocs
        corpus1[i] = create_single_doc(doclens1[i], Theta1[i],B1, Terms1)
        corpus2[i] = create_single_doc(doclens2[i], Theta2[i],B2, Terms2)
    end
    return (corpus1,corpus2, Theta, B1, B2, Terms1, Terms2)
end
