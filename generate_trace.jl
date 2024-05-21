using Revise
using AlphaZero
using Statistics
using GLMakie, GraphMakie
using JSON
using SparseArrayKit
using SparseArrays
using LightGraphs

function gen_graph(n_nodes::Int, target_edges::Int)::LightGraphs.AbstractGraph

  # Start with an Erdős-Rényi graph with a slightly higher edge probability
  p = target_edges / binomial(n_nodes, 2)
  g = LightGraphs.SimpleGraphs.erdos_renyi(n_nodes, p)

  # Adjust the number of edges to the target by adding or removing edges
  while true
    g = LightGraphs.SimpleGraphs.erdos_renyi(n_nodes, p)
    while LightGraphs.ne(g) != target_edges
      if LightGraphs.ne(g) > target_edges
        # Remove random edges if there are too many
        edges = collect(LightGraphs.edges(g))
        random_edge = rand(edges)
        u, v = Tuple(random_edge)
        LightGraphs.rem_edge!(g, u, v)
      elseif LightGraphs.ne(g) < target_edges
        # Add edges if there are too few
        possible_edges = [(i, j) for i in 1:n_nodes for j in i+1:n_nodes if !LightGraphs.has_edge(g, i, j)]
        LightGraphs.add_edge!(g, rand(possible_edges))
      end
    end
    if LightGraphs.is_connected(g)
      break
    end
  end



  return g
end

global EVALMODE = true

function read_json_file(filename)
  open(filename, "r") do f
    return JSON.parse(f)
  end
end

function parse_tuple(s::String)
  s = replace(s, "(" => "", ")" => "", " " => "")
  return Tuple(parse(Int, x) for x in split(s, ","))
end


data = read_json_file("25_tensors_3mean.json")

rewards = []
times = []

for i in 76:100
  print("\n\n ##########  ", i, "  ########## \n\n")
  start = time()
  str_dict = data[i]["eq_$(i-1)"]


  tuple_dict = Dict(parse_tuple(k) => v for (k, v) in str_dict)
  I = [i[1] + 1 for (i, v) in tuple_dict]
  J = [i[2] + 1 for (i, v) in tuple_dict]
  V = [v for (i, v) in tuple_dict]
  sparse(I, J, V)


  global eval_A = SparseArray(sparse(I, J, V))
  global eval_sizes = reshape(data[i]["eq_size_$(i-1)"], 1, :)

  #   I = [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5]
  #   J = [2, 5, 1, 3, 2, 4, 5, 3, 5, 1, 3, 4]
  #   V = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  #   N = 9
  #   S = 12
  #   A = LightGraphs.adjacency_matrix(gen_graph(N, S))
  #   print(A)
  #   global eval_A = SparseArray(sparse(I, J, V))
  #   global eval_sizes = reshape([2 for h in 1:6], 1, :)
  global nedges = size(eval_sizes)[2]
  print(nedges)


  include("params/params_N25_S34_Networktest_25_5_competing.jl")

  include("environments/tensor_alphazeroRTN_greedy_competing_penalty.jl")
  experiment = AlphaZero.Experiment("testing2", GameSpec(), params, Network, netparams, benchmark)


  Ses = Session(experiment)
  gspec = Ses.env.gspec
  print(Ses.env.params.arena)

  p = AlphaZeroPlayer(Ses.env, timeout=0.5)
  print(p.niters)
  #p = NetworkPlayer(AlphaZero.Network.copy(Ses.env.bestnn, on_gpu=false, test_mode=true))
  for j in 1:1
    global trace = play_game(gspec, p; flip_probability=0.0, id=1, worker_id=1, reset_every=nothing)
    print(last(trace.states).greedy_result, "\n")
    print(last(trace.states).history, last(trace.states).total_reward * last(trace.states).greedy_result, "\n")
    if j == 1
      global best = last(trace.states).total_reward * last(trace.states).greedy_result
    else
      global best = max(best, last(trace.states).total_reward * last(trace.states).greedy_result)
    end
  end
  print(best, "\n")
  push!(rewards, best)
  push!(times, time() - start)
end
print(rewards, "\n")
print(times, "\n")
print(mean(rewards))

#trace = play_game(gspec, p; flip_probability=0.)

# function plotgraph(fg)
#   print([e for e in GraphSignals.edges(fg)][1:GraphSignals.ne(fg)])
#   S = fg.graph.S

#   g = Graphs.SimpleGraph(Matrix(S))

#   nlabels = [string(Char(letter)) for letter in UInt32('A'):(UInt32('A')-1+Graphs.nv(g))]
#   elabels = [string(collect(edge_feature(fg))[:,i]) for i in unique(collect(GraphSignals.edges(fg))[1])]

#   fig, ax, p = graphplot(g, node_color=:lightblue, node_size =[40 for i in 1:Graphs.nv(g)], nlabels=nlabels,edge_width=[1 for i in 1:Graphs.ne(g)],  nlabels_align = (:center, :center), elabels = elabels, markersize=15, color=:black)
#   hidedecorations!(ax)


#   deregister_interaction!(ax, :rectanglezoom)
#   register_interaction!(ax, :nhover, NodeHoverHighlight(p))
#   register_interaction!(ax, :ehover, EdgeHoverHighlight(p))
#   register_interaction!(ax, :ndrag, NodeDrag(p))
#   register_interaction!(ax, :edrag, EdgeDrag(p))
#   return fig
# end





# for fg in fgs
#     fig = plotgraph(fg)
#     display(fig)
#     println("Press Enter to continue to the next plot...")
#     readline() 
# end

