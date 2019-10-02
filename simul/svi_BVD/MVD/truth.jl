using Random
Random.seed!(1234)
using DataFrames
using DelimitedFiles
using ArgParse
using FileIO
using JLD2
using BenchmarkTools
using Logging
include("utils.jl")
include("init.jl")
include("MVD.jl")
include("dgp.jl")
include("funcs.jl")
function main(args)
	s = ArgParseSettings()
    @add_arg_table s begin
		"-n"
            help = "number of docs"
            arg_type=Int64
            default=2000
		"--k1"
            help = "number of topics in mode 1"
            arg_type=Int64
            default=5
		"--k2"
            help = "number of topics in mode 2"
            arg_type=Int64
            default=5
		"--v1"               #number of communities
            help = "number of vocabs in mode 1"
            arg_type=Int64
            default=200
		"--v2"               #number of communities
            help = "number of vocabs in mode 2"
            arg_type=Int64
            default=200
		"--wlen1"
            help = "number of words per doc 1"
            arg_type=Int64
            default=200
		"--wlen2"
            help = "number of words per doc 2"
            arg_type=Int64
            default=200
		"--alpha"
			help = "alpha truth"
			arg_type = Float64
			default = .3
		"--beta1"
			help = "beta1 truth"
			arg_type = Float64
			default = .3
		"--beta2"
			help = "beta2 truth"
			arg_type = Float64
			default = .3
		"--sparse"
			help = "sparsity in mode 2"
			arg_type = Float64
			default = 0.0

    end
    # # #
    parsed_args = ArgParse.parse_args(args,s) ##result is a Dict{String, Any}
    @info "Parsed args: "
    for (k,v) in parsed_args
        @info "  $k  =>  $(repr(v))"
    end
    @info "before parsing"

	N = parsed_args["n"]
	K1 = parsed_args["k1"]
	K2 = parsed_args["k2"]
	V1 = parsed_args["v1"]
	V2 = parsed_args["v2"]
	α_single_truth = parsed_args["alpha"]
	β1_single_truth = parsed_args["beta1"]
	β2_single_truth = parsed_args["beta2"]
	wlen1_single = parsed_args["wlen1"]
	wlen2_single = parsed_args["wlen2"]
	sparsity = parsed_args["sparse"]



	N = 2000
	K1 = 5
	K2 = 3
	V1 = 200
	V2 = 100
	α_single_truth = 0.1
	β1_single_truth = .3
	β2_single_truth = .3
	wlen1_single = 100
	wlen2_single = 80
	sparsity = 0.3


	folder = mkdir("$(N)_$(K1)_$(K2)_$(V1)_$(V2)_$(α_single_truth)_$(β1_single_truth)_$(β2_single_truth)_$(sparsity)")
	#########################
	α,Α, θ,Θ, Β1, Β2, β1, β2, V1, V2, corp1, corp2 =
	 Create_Truth(N, K1, K2, V1, V2, β1_single_truth, β2_single_truth, wlen1_single, wlen2_single, sparsity)

	 α_truth,Α_truth, θ_truth,Θ_truth,Β1_truth, Β2_truth, β1_truth, β2_truth,V1, V2, corp1, corp2=
	 simulate_data(N, K1, K2, V1, V2,β1_single_truth, β2_single_truth,wlen1_single, wlen2_single,
	 sparsity)

	 # B11 = deepcopy(collect(transpose(Β1_truth)))
	 # B11 = collect(transpose((sort_by_argmax!(B11))[1]))
	 # B22 = deepcopy(collect(transpose(Β2_truth)))
	 # B22 = collect(transpose((sort_by_argmax!(B22))[1]))
	 #
	 # Plots.heatmap(B11, yflip=true)
	 # Plots.heatmap(B22, yflip=true)



	Truth_Params = Params(N,K1,K2,V1,V2,α_truth,Α_truth,θ_truth,Θ_truth,β1_truth,β2_truth,Β1_truth,Β2_truth)
	@save "$(folder)/truth" Truth_Params
	Docs1 = [Document(corp1[i], corp1[i], length(corp1[i])) for i in 1:length(corp1)]
	Docs2 = [Document(corp2[i], corp2[i], length(corp2[i])) for i in 1:length(corp2)]



	Corpus1 = Corpus(Docs1, length(Docs1), V1)
	Corpus2 = Corpus(Docs2, length(Docs2), V2)
	@save "$(folder)/corpus1" Corpus1
	@save "$(folder)/corpus2" Corpus2
end

main(ARGS)
