using Distributions
using DataStructures
using StatsBase
using StatsPlots
using Plots
# using DataFrames
function simulate_LDA(ndocs, nterms, K, alpha, beta, doclens)

    Alpha = repeat([alpha], K)
    Beta = repeat([beta], nterms)

    #creating labels
    Terms = ["Term$i" for i in 1:nterms]
    Topics = ["Topic$k" for k in 1:K]
    Documents = ["Doc$d" for d in 1:ndocs]

    Theta = transpose(rand(Distributions.Dirichlet(Alpha),ndocs))
    B = transpose(rand(Distributions.Dirichlet(Beta), K))

    function create_single_doc(doclen, topic_dist, term_topic_dist, Terms)
        doc = String[]
        for i in 1:doclen
            topic = argmax(rand(Distributions.Multinomial(1,topic_dist)))
            term =  argmax(rand(Distributions.Multinomial(1,term_topic_dist[topic,:])))
            doc = vcat(doc, Terms[term])
        end
        return doc
    end
    corpus = [[] for i in 1:ndocs]
    for i in 1:ndocs
        corpus[i] = create_single_doc(doclens[i], Theta[i,:],B, Terms)
    end
    return corpus
end


alpha=.1
beta=0.01
ndocs = 250
nterms = 2000
K=5
doclens = repeat([100], ndocs)

corpus = simulate_LDA(ndocs, nterms, K, alpha, beta, doclens)
