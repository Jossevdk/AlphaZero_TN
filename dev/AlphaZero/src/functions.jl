import CUDA
using SparseArrays
import GeometricFlux: GraphSignals

function SparseArrays.findnz(S::CUDA.CUSPARSE.CuSparseMatrixCSC) 
  S2 = CUDA.CUSPARSE.CuSparseMatrixCOO(S)
  I = S2.rowInd
  J = S2.colInd
  V = S2.nzVal

  # To make it compatible with the SparseArrays.jl version
  idxs = sortperm(J)
  I = I[idxs]
  J = J[idxs]
  V = V[idxs]

  return (I, J, V)
end

#Edges are ordered when first looping over the list of lists 
#and then over this list only counted once with taking the upper triangle, 
#with the index of a column equal to the index of a list in the list of lists

mutable struct BatchFeatureGraph
    fg::FeaturedGraph
    graph_indicator::Union{Vector{Int64}, CUDA.CuArray}
    graph_indicator_v::Union{Vector{Int64}, CUDA.CuArray}
    function BatchFeatureGraph(fg::FeaturedGraph, graph_indicator::Union{Vector{Int64}, CUDA.CuArray}, graph_indicator_v::Union{Vector{Int64}, CUDA.CuArray})
        new(fg, graph_indicator, graph_indicator_v)
    end
end
fg(fbg::BatchFeatureGraph) = fbg.fg

function Flux.gpu(bfg::BatchFeatureGraph)
    return BatchFeatureGraph(Flux.gpu(bfg.fg), bfg.graph_indicator, bfg.graph_indicator_v)
end

function Flux.batch(fgs::AbstractVector{<:FeaturedGraph})

    v_num_nodes = [GraphSignals.nv(fg) for fg in fgs]
    nodesum = cumsum([0; v_num_nodes])[1:(end - 1)]
    rows = vcat([findnz(fg.graph.S)[1] .+ nodesum[i] for (i, fg) in enumerate(fgs)]...)
    cols = vcat([findnz(fg.graph.S)[2] .+ nodesum[i] for (i, fg) in enumerate(fgs)]...)
    vals = vcat([findnz(fg.graph.S)[3] for fg in fgs]...)
    # print(rows, cols, vals)
    S = sparse(rows, cols, vals)
    ef = hcat([edge_feature(fg) for fg in fgs]...)
    nf = hcat([node_feature(fg) for fg in fgs]...)
    
    #nf = hcat([fg.nf for fg in fgs]...)

    function materialize_graph_indicator(fg)
        ones(Int64, GraphSignals.ne(fg))
    end

    e_gi = materialize_graph_indicator.(fgs)
    
    graphsum = 0:(length(fgs)-1)
    e_gi = [ng .+ gi for (ng, gi) in zip(graphsum, e_gi)]
    graph_indicator = vcat(e_gi...)

    function materialize_graph_indicator_v(fg)
        ones(Int64, GraphSignals.nv(fg))
    end

    v_gi = materialize_graph_indicator_v.(fgs)
    
    graphsum_v = 0:(length(fgs)-1)
    v_gi = [ng .+ gi for (ng, gi) in zip(graphsum, v_gi)]
    graph_indicator_v = vcat(v_gi...)
    

    return BatchFeatureGraph(FeaturedGraph(S, nf= nf, ef = ef), graph_indicator, graph_indicator_v)
end
function Base.copy(fg::FeaturedGraph)
    return FeaturedGraph(fg)
end

# Define hash function for FeaturedGraph
function Base.hash(fg::FeaturedGraph, h::UInt)
    return hash(fg.graph, hash(fg.ef.signal, hash(fg.nf.signal, h)))
end

function Base.isequal(fg::FeaturedGraph, fh::FeaturedGraph)
    return fg.graph == fh.graph && fg.ef.signal == fh.ef.signal && fg.nf.signal == fh.nf.signal
end

function Base.:(==)(fg::FeaturedGraph, fh::FeaturedGraph)
    return isequal(fg, fh)
end




function Base.hash(sg::SparseGraph, h::UInt)
    return hash(sg.S,  hash(sg.edges, h))
end

function Base.isequal(sg::SparseGraph, sh::SparseGraph)
    return sg.S == sh.S && sg.edges == sh.edges && sg.E == sh.E
end



function get_edges_ary(con_ary::Array{Int64, 2})
    edges_ary = zeros(Int, size(con_ary))
    
    edges_ary[con_ary .== -1] .= -1
  
    num_edges = 1
    for i in axes(con_ary, 1)
        for j in axes(con_ary, 2)
            if con_ary[i, j] == -1
                continue
            end
            if i == con_ary[i, j]
                edges_ary[i, i] = num_edges
                num_edges +=1
            end
            if i>con_ary[i, j]
                edges_ary[i, j] = num_edges
                edges_ary[con_ary[i, j], findfirst(x -> x == i, con_ary[con_ary[i, j],:])] = num_edges
                num_edges +=1
            end
        end
    end
    #print(typeof(edges_ary))
    return edges_ary
end

function get_edges_ary(nodes_ary::Array{Int64, 3})
    con_ary = nodes_ary[:, :, 1]
    return get_edges_ary(con_ary)
end

function get_node_dims_arys(con_ary::Array{Int64, 2}, e_weight::Array{Int64, 2})
    num_nodes = size(con_ary,1)


    arys = Vector{Int}[]
    for (i, nodes) in enumerate(eachrow(con_ary))
        ary = zeros(Int, num_nodes) 
        for (j, node) in enumerate(nodes)
            positive_nodes = nodes[nodes .>= 0]
            if con_ary[i, j] >0
                ary[node] =  e_weight[i, j] 
            end
        end
        push!(arys, ary)
        
    end
    
    return stack(arys)
end



function get_node_bool_arys(con_ary::Array{Int64, 2})
    num_nodes = size(con_ary,1)

    arys = Vector{Bool}[]
    for (i, nodes) in enumerate(eachrow(con_ary))
        ary = zeros(Bool, num_nodes) 
        ary[i] = true
        
        push!(arys, ary)
    end
    
    return stack(arys)
end

function matrixToAdjList(matrix)
    # Initialize an empty list of vectors to store the adjacency list
    adjList = Vector{Int}[]

    # Iterate through each row of the matrix
    for row in eachrow(matrix)
        # Filter out '-1' and add the remaining elements as a vector to the adjList
        push!(adjList, [element for element in row if element != -1])
    end

    return adjList


end
#Edge-weights are ordered when first looping over the list of lists 
#and then over this list only counted once with taking the upper triangle, 
#with the index of a column equal to the index of a list in the list of lists
function esizeFromNodesAry(con_ary, e_weigth)
    
    esize = Float32[]
    
    for i in axes(con_ary, 1)
        for j in axes(con_ary, 2)
            if i>con_ary[i, j] && con_ary[i, j] !== -1
                push!(esize, e_weigth[i, j])
            end
            
        end
    end
    return esize'



end
function adjListToSparseAdjMatrix(adjacency_list)
    rows = Int[]
    cols = Int[]
    vals = Int[]

    # Fill the row indices, column indices, and values
    for (i, neighbors) in enumerate(adjacency_list)
        for neighbor in neighbors
            push!(rows, neighbor)
            push!(cols, i)
            push!(vals, 1)  # Using 1 for unweighted graphs
        end
    end

    return sparse(rows, cols, vals)
    
end

function get_nodes_from_edge(sg, edge)
    (e, r, _) = collect(GraphSignals.edges(sg))
    i, j = findall(x -> x == edge, e)
    return r[i], r[j]
end

function get_edge_from_nodes(sg, i, j)
    (e, r, c) = collect(GraphSignals.edges(sg))
    j = findfirst(x -> x== (i,j), [(r[i], c[i]) for i in 1:length(r)])
    return e[j]
end



"""
    edge_betweenness_centrality(graph::SparseGraph)

Calculate the edge betweenness centrality for each edge in a sparse graph.
"""
function edge_betweenness_centrality(graph::SparseGraph)
    n = size(graph.S, 1)  # number of vertices
    edge_centrality = Dict{Tuple{Int, Int}, Float64}()

    # Iterate over all vertices to use each one as a source
    for s in 1:n
        # Structures for Brandes' algorithm
        S = Int[]
        P = [Int[] for _ in 1:n]
        sigma = zeros(Int, n)
        sigma[s] = 1
        d = fill(-1, n)
        d[s] = 0
        Q = Int[]
        push!(Q, s)

        # BFS to calculate shortest paths
        while !isempty(Q)
            v = popfirst!(Q)
            push!(S, v)
            for w in neighbors(graph, v)
                if d[w] < 0
                    push!(Q, w)
                    d[w] = d[v] + 1
                end
                if d[w] == d[v] + 1
                    sigma[w] += sigma[v]
                    push!(P[w], v)
                end
            end
        end

        # Back-propagation to calculate edge centrality
        delta = zeros(Float64, n)
        while !isempty(S)
            w = pop!(S)
            for v in P[w]
                c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                edge_centrality[(v, w)] = get(edge_centrality, (v, w), 0.0) + c
                edge_centrality[(w, v)] = get(edge_centrality, (w, v), 0.0) + c  # For undirected graphs, add this line
                delta[v] += c
            end
        end
    end

    # Normalize the centrality scores
    for key in keys(edge_centrality)
        edge_centrality[key] /= 2
    end

    return edge_centrality
end
