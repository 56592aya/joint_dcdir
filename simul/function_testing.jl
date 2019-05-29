using SpecialFunctions
#how does digamma function look like for different positive values

xrange = LinRange(0.1, 10.0, 100)
Plots.plot(xrange, [lgamma_(x) for x in xrange])
Plots.plot(xrange, [lgamma(x) for x in xrange])
Plots.plot(xrange, [digamma_(x) for x in xrange])
Plots.plot(xrange, [digamma(x) for x in xrange])
Plots.plot(xrange, [trigamma_(x) for x in xrange])
Plots.plot(xrange, [trigamma(x) for x in xrange])

funclist1 = [lgamma, digamma, trigamma]
funclist2 = [lgamma_, digamma_, trigamma_]
vals = [0.0001, 0.005, 0.01, 0.05, 0.1, .5, 1.0, 1.5, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]

for val in vals
    # println("lgamma for val = $val  :   $(funclist1[1](val)) , and $(funclist2[1](val))")
    x = (funclist1[2](val)) - (funclist2[2](val))
    println("digamma for val = $val  :  $x ")
    # println("trigamma for val = $val  :   $(funclist1[3](val)) , and $(funclist2[3](val))")
end


lgamma(5000)
println(" For small values lgamma of SpecialFunctions seems safer and better behaved")

function draw_dir(K, a_scalar)
    draw_dir_diff(a_scalar.*ones(Float64, K))
end
function draw_dir_diff(a_vec)
    K = length(a_vec)
    ρ = Distributions.rand(Distributions.Dirichlet(a_vec), 1)
    Plots.bar(ρ,xlims=(0,K+1), ylims=(0,1), ylabel="freqs", xlabel="index", color="red",
    title="a_vec = $a_vec")
end
K=500
a_scalar = 1.0/(K^(0.8))
draw_dir(K,a_scalar)
a_vec = normalize([a_scalar for k in 1:K],1)
draw_dir_diff(a_vec)


# K=500
# a_vec = a_scalar.*ones(Float64, K)
# K = length(a_vec)
# ρ = Distributions.rand(Distributions.Dirichlet(a_vec), 1)[:,1]
# Ρ = convert(Matrix{Float64}, transpose(reshape(ρ, (25, 20))))
# Plots.heatmap(Ρ)
