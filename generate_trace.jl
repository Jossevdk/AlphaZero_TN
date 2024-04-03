include("tensor_alphazero3GF.jl")
include("params_test.jl")
include("functions.jl")

# nodes_ary = [2 5 -1; 1 3 9; 2 7 -1; 5 8 -1; 1 4 6; 5 7 -1; 3 6 8; 4 7 9; 2 8 -1;;; 4 4 -1; 4 4 4; 4 4 -1; 4 4 -1; 4 4 4; 4 4 -1; 4 4 4; 4 4 4; 4 4 -1]
# edges = get_edges_ary(nodes_ary)


Ses = Session(experiment)
p = AlphaZeroPlayer(Ses.env)
gspec = Ses.env.gspec

trace = play_game(gspec, p; flip_probability=0.)

print("\n\n\n________________\n")
print(last(trace.states).history, trace.rewards)
