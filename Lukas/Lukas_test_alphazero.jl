using Distributed
using Profile


print(nworkers())
# @everywhere using Pkg
# @everywhere Pkg.develop(path = "/home/josse/Documents/thesis/github_AZTN/AlphaZero_TN/dev/EinExprs")
# @everywhere Pkg.develop(path = "/home/josse/Documents/thesis/github_AZTN/AlphaZero_TN/dev/AlphaZero")
# @everywhere Pkg.activate(".")
@everywhere using Revise
@everywhere using AlphaZero




@everywhere global EVALMODE = false
@everywhere include("params_N25_S34_Networktest_25_5_score_wins_arena.jl")
@everywhere include("tensor_alphazeroRTN_greedy_averaged.jl")

experiment = AlphaZero.Experiment("Lukas_testing1_wins_arena", GameSpec(), params, Network, netparams, benchmark)



#Scripts.test_game(experiment)
Profile.init(n = 10^8, delay = 0.1)
Ses = Session(experiment) 
resume!(Ses)
