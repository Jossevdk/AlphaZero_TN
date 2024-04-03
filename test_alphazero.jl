using AlphaZero

#using Distributed
#
#addprocs(2)
##mypath is the path the module file
#@everywhere push!(LOAD_PATH,$"dev")
#@everywhere using AlphaZero

include("tensor_alphazero3GF.jl")
include("params_test.jl")

#Scripts.test_game(experiment)
Ses = Session(experiment)
resume!(Ses)

