using GLMakie, GraphMakie,  Graphs
using NetworkLayout

include("/home/josse/Documents/thesis/Alpha_Zero_TN/functions.jl")

nodes_ary = [2 -1 -1 -1;1 3 4 5; 2 5 -1 -1; 2 5 -1 -1;2 3 4 6; 5 -1 -1 -1;;;
             2 -1 -1 -1;2 2 2 2; 2 2 -1 -1; 2 2 -1 -1;2 2 2 2; 2 -1 -1 -1]

#nodes_ary = [5 13 14 -1 -1 -1 -1 -1; 
#3 4 7 12 15 18 -1 -1; 
#2 7 11 -1 -1 -1 -1 -1; 
#2 7 10 14 15 -1 -1 -1; 
#1 7 8 11 12 17 20 -1; 
#8 12 13 -1 -1 -1 -1 -1; 
#2 3 4 5 10 16 17 18; 
#5 6 12 -1 -1 -1 -1 -1; 
#11 14 -1 -1 -1 -1 -1 -1; 
#4 7 19 -1 -1 -1 -1 -1; 
#3 5 9 12 15 17 18 -1; 
#2 5 6 8 11 14 17 -1; 
#1 6 19 20 -1 -1 -1 -1; 
#1 4 9 12 15 16 -1 -1; 
#2 4 11 14 16 -1 -1 -1; 
#7 14 15 20 -1 -1 -1 -1; 
#5 7 11 12 18 -1 -1 -1; 
#2 7 11 17 -1 -1 -1 -1; 
#10 13 20 -1 -1 -1 -1 -1; 
#5 13 16 19 -1 -1 -1 -1;;; 
#9 4 13 -1 -1 -1 -1 -1; 
#2 4 3 3 8 2 -1 -1; 
#2 4 9 -1 -1 -1 -1 -1; 
#4 3 11 5 9 -1 -1 -1; 
#9 9 5 4 6 3 2 -1; 
#4 5 2 -1 -1 -1 -1 -1; 
#3 4 3 9 4 3 6 3; 
#5 4 4 -1 -1 -1 -1 -1; 
#8 9 -1 -1 -1 -1 -1 -1; 
#11 4 4 -1 -1 -1 -1 -1; 
#9 4 8 8 8 8 11 -1; 
#3 6 5 4 8 8 5 -1; 
#4 2 4 3 -1 -1 -1 -1; 
#13 5 9 8 5 9 -1 -1; 
#8 9 8 5 8 -1 -1 -1; 
#3 9 8 2 -1 -1 -1 -1; 
#3 6 8 5 6 -1 -1 -1; 
#2 3 11 6 -1 -1 -1 -1; 
#4 4 9 -1 -1 -1 -1 -1; 
#2 3 2 9 -1 -1 -1 -1]


node_dims_arys = stack(get_node_dims_arys(nodes_ary))



adj_matrix = copy(node_dims_arys)
adj_matrix[adj_matrix .!= 0] .= 1
weights = node_dims_arys




g = Graphs.SimpleGraph(adj_matrix) # Use SimpleGraph 



# Start plotting

nlabels = [string(Char(letter)) for letter in UInt32('A'):(UInt32('A')-1+nv(g))]
elabels = [string(weights[Graphs.src(e), Graphs.dst(e)]) for e in Graphs.edges(g)]

fig, ax, p = graphplot(g, node_color=:lightblue, node_size =[40 for i in 1:nv(g)], nlabels=nlabels,edge_width=[1 for i in 1:ne(g)],  nlabels_align = (:center, :center), elabels = elabels, markersize=15, color=:black)
hidedecorations!(ax)


deregister_interaction!(ax, :rectanglezoom)
register_interaction!(ax, :nhover, NodeHoverHighlight(p))
register_interaction!(ax, :ehover, EdgeHoverHighlight(p))
register_interaction!(ax, :ndrag, NodeDrag(p))
register_interaction!(ax, :edrag, EdgeDrag(p))


# Display the figure
fig