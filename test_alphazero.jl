using Distributed
@everywhere using Revise
@everywhere using AlphaZero

#using Distributed
#
#addprocs(2)
##mypath is the path the module file
#@everywhere push!(LOAD_PATH,$"dev")
#@everywhere using AlphaZero

@everywhere include("tensor_alphazero3RTN.jl")
@everywhere include("params_test.jl")

#Scripts.test_game(experiment)
Ses = Session(experiment) 
resume!(Ses)

