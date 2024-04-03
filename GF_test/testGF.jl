using GeometricFlux
using Graphs
using SparseArrays
using GLMakie, GraphMakie
using NetworkLayout
#using CUDA
import GraphSignals
import Flux

include("/home/josse/Documents/thesis/Alpha_Zero_TN/functions.jl")
g = [[2, 8, 9, 10], [1, 3, 5, 6], [2, 7, 10], [5, 6], [2, 4], [2, 4, 8], [3], [1, 6, 10], [1], [1, 3, 8]]
h = [[2,3, 6], [1,4,5], [1], [2,5], [2,4, 6], [5, 1, 7], [6]]

amask = trues(6)

A = sparse([1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5], [2, 5, 1, 3, 5, 2, 4, 3, 5, 1, 2, 4], [true, true, true, true, true, true, true, true, true, true, true, true])
print(typeof(A))
ef = [4 256; 4 256; 4 64; 4 256; 4 1024; 4 256]

fg = FeaturedGraph(A, ef=ef')

o_edge = collect(1:ne(fg))

function update(fg, o_edge, amask, action)
  ef = collect(edge_feature(fg))
  ef = [c[:] for c in eachcol(ef)]
  print(ef)
  print([e for e in edges(fg)][1:ne(fg)])
  A = copy(fg.graph.S)
  es, nbrs, xs = collect(edges(fg.graph))



  
  edge = action #- sum(.!amask[1:action])
  n1i, n2i = get_nodes_from_edge(fg.graph, edge)
  print(n1i, " , ", n2i, "\n")

  last_true_index = findlast(x -> x == true, amask)
  amask[last_true_index] = false
  A[n1i, n2i] = false
  A[n2i, n1i] = false
  dropzeros!(A)

  delete =[edge]
  for i in enumerate(A[n1i,:])
      if i[2] && A[n2i, i[1]]
          j = get_edge_from_nodes(fg.graph, i[1], n2i)
          k = get_edge_from_nodes(fg.graph, i[1], n1i)
          last_true_index = findlast(x -> x == true, amask)
          amask[last_true_index] = false
          push!(delete, j)
          ef[k][1] *= ef[j][1] 


        end
      A[n2i, i[1]] = i[2]||A[n2i, i[1]]
      A[i[1], n2i] = i[2]||A[i[1], n2i] 
      A[n1i, i[1]] = false
      A[i[1], n1i] = false
  end
  
  print(amask)

  A = A[setdiff(1:end, n1i), setdiff(1:end, n1i)]

  setdiff!(o_edge, delete)
  deleteat!(ef, sort(delete))
  print("\n =========== \n")
  print(ef)
  edg = [e for e in edges(fg)][1:ne(fg)]
  print("\n =========== EDG \n")
  print(edg)
  deleteat!(edg, sort(delete))
  new_edg =[]
  for ed in edg
      (e, (n1, n2)) = ed
      if n1 ==n1i
          n1 = n2i
      end
      if n2 == n1i
          n2 = n2i
      end
      if n1 > n2
          push!(new_edg, (e, (n1, n2)))
      else
          push!(new_edg, (e, (n2, n1)))
      end

  end
  s = sortperm([e[2] for e in new_edg])

  print(new_edg[s])
  ef = ef[s]
  print(ef)

  
    
  o_edge = o_edge[s]
  print("\n =========== NEW \n")
  
  print(ef, " , ", o_edge, "\n")
  fg = FeaturedGraph(A, ef=hcat(ef...))
  edg = [e for e in edges(fg)][1:ne(fg)]
  update_e = incident_edges(fg.graph, n2i)
  for i in update_e
    (e, (n1, n2)) = edg[i]
    indices = [incident_edges(fg.graph, n1); incident_edges(fg.graph, n2)]
    print(indices)
    ef[e][2] = prod([ef[i][1] for i in indices])/ef[e][1]
  end
  fg.ef = hcat(ef...)
  print([e for e in edges(fg)][1:ne(fg)])
  return fg, o_edge, amask
end










function plotgraph(fg)
  print([e for e in edges(fg)][1:ne(fg)])
  S = fg.graph.S

  g = Graphs.SimpleGraph(Matrix(S))

  nlabels = [string(Char(letter)) for letter in UInt32('A'):(UInt32('A')-1+nv(g))]
  elabels = [string(collect(edge_feature(fg))[:,i]) for i in unique(collect(edges(fg))[1])]

  fig, ax, p = graphplot(g, node_color=:lightblue, node_size =[40 for i in 1:nv(g)], nlabels=nlabels,edge_width=[1 for i in 1:ne(g)],  nlabels_align = (:center, :center), elabels = elabels, markersize=15, color=:black)
  hidedecorations!(ax)


  deregister_interaction!(ax, :rectanglezoom)
  register_interaction!(ax, :nhover, NodeHoverHighlight(p))
  register_interaction!(ax, :ehover, EdgeHoverHighlight(p))
  register_interaction!(ax, :ndrag, NodeDrag(p))
  register_interaction!(ax, :edrag, EdgeDrag(p))
  fig
end

Flux.gpu(fg)

bfg = 
fh = copy(fg)
print( fg, edge_feature(fg))
print(fh, edge_feature(fh))
print(fg == fh)
fg, o_edge, amask= update(fg, o_edge, amask, 1)
#plotgraph(fg)

fg, o_edge, amask = update(fg, o_edge, amask, 3)

#plotgraph(fg)
myDict = Dict(fg => "value1", fh => "value2")

print("\n", haskey(myDict, fg))
print("\n", haskey(myDict, copy(fg)))
