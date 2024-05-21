using Distributed
using Profile


print(nworkers())
# @everywhere using Pkg
# @everywhere Pkg.develop(path = "/home/josse/Documents/thesis/github_AZTN/AlphaZero_TN/dev/EinExprs")
# @everywhere Pkg.develop(path = "/home/josse/Documents/thesis/github_AZTN/AlphaZero_TN/dev/AlphaZero")
# @everywhere Pkg.activate(".")
@everywhere using Revise
@everywhere using AlphaZero



@everywhere global iteration = 1
@everywhere global EVALMODE = false
for it in 1:5
    GC.gc()
    global iteration = it
    @everywhere include("params/params_N25_S34_Networktest_25_5_score_wins_arena.jl")
    @everywhere include("environments/tensor_alphazeroRTN.jl")

    experiment = AlphaZero.Experiment("testing1_no_average", GameSpec(), params, Network, netparams, benchmark)



    #Scripts.test_game(experiment)
    Profile.init(n=10^8, delay=0.1)
    Ses = Session(experiment)
    resume!(Ses)
    print("\n\n\n ################ITERATION ENDED################## \n\n\n")
    sleep(2)
    GC.gc()

    
end
