using DataStructures
using DataFrames
using Distributions
using DelimitedFiles
using JLD2
using FileIO
using StatsBase
using StatPlots
using Plots
import Base: ==, hash, isequal, zeros
import Core: ===
using LightGraphs
using MetaGraphs
using ArgParse
using CSV
using DoWhile
using Missings
using GradDescent
using SparseArrays
using LinearAlgebra
using Random
using StatProfilerHTML

###types

VectorList{T} = Vector{Vector{T}}
MatrixList{T} = Vector{Matrix{T}}
Matrix2d{T}   = Matrix{T}
Matrix3d{T}   = Array{T,3}
Network{T}    = SparseMatrixCSC{T,T}

Map{R,T} = Dict{R,T}

mutable struct KeyVal
  first::Int64
  second::Float64
end
