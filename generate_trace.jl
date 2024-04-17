include("tensor_alphazero3RTN.jl")
include("params_test.jl")
include("functions.jl")
using GLMakie, GraphMakie

# nodes_ary = [2 5 -1; 1 3 9; 2 7 -1; 5 8 -1; 1 4 6; 5 7 -1; 3 6 8; 4 7 9; 2 8 -1;;; 4 4 -1; 4 4 4; 4 4 -1; 4 4 -1; 4 4 4; 4 4 -1; 4 4 4; 4 4 4; 4 4 -1]
# edges = get_edges_ary(nodes_ary)


Ses = Session(experiment)
p = AlphaZeroPlayer(Ses.env)
p = RandomPlayer()
gspec = Ses.env.gspec

trace = play_game(gspec, p; flip_probability=0.)
fgs = [s.fg for s in trace.states]

function plotgraph(fg)
  print([e for e in GraphSignals.edges(fg)][1:GraphSignals.ne(fg)])
  S = fg.graph.S

  g = Graphs.SimpleGraph(Matrix(S))

  nlabels = [string(Char(letter)) for letter in UInt32('A'):(UInt32('A')-1+Graphs.nv(g))]
  elabels = [string(collect(edge_feature(fg))[:,i]) for i in unique(collect(GraphSignals.edges(fg))[1])]

  fig, ax, p = graphplot(g, node_color=:lightblue, node_size =[40 for i in 1:Graphs.nv(g)], nlabels=nlabels,edge_width=[1 for i in 1:Graphs.ne(g)],  nlabels_align = (:center, :center), elabels = elabels, markersize=15, color=:black)
  hidedecorations!(ax)


  deregister_interaction!(ax, :rectanglezoom)
  register_interaction!(ax, :nhover, NodeHoverHighlight(p))
  register_interaction!(ax, :ehover, EdgeHoverHighlight(p))
  register_interaction!(ax, :ndrag, NodeDrag(p))
  register_interaction!(ax, :edrag, EdgeDrag(p))
  return fig
end



print("\n\n\n________________\n")
print( last(trace.states).history, sum(trace.rewards))

# for fg in fgs
#     fig = plotgraph(fg)
#     display(fig)
#     println("Press Enter to continue to the next plot...")
#     readline() 
# end

