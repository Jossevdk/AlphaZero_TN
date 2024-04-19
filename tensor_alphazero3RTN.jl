using AlphaZero
using StaticArrays
using SparseArrays
import AlphaZero.GI
using Base: @kwdef
using GeometricFlux
using GraphSignals
using Graphs
using LightGraphs
import Flux
using Flux: batch
using SparseArrayKit



include("functions.jl")

global E = 0
global S = 0
global N = 0


struct GameSpec <: GI.AbstractGameSpec end

"""
Setup as GameEnv.
In this environment the state is a adjecancy matrix.
"""
mutable struct GameEnv <: GI.AbstractGameEnv
  fg::AbstractFeaturedGraph
  reward::Float32
  o_edge::Vector{Int}
  amask::BitVector
  finished::Bool
  history :: Union{Nothing, Vector{Int}}
    
end


GI.spec(::GameEnv) = GameSpec()

function GI.init(::GameSpec)
  #con_ary = [2 8 9 10; 1 3 5 6; 2 7 10 -1; 5 6 -1 -1; 2 4 -1 -1; 2 4 8 -1; 3 -1 -1 -1; 1 6 10 -1; 1 -1 -1 -1; 1 3 8 -1]
  #e_weight =  [9 7 6 5; 9 5 4 4; 5 11 9 -1; 9 9 -1 -1; 4 9 -1 -1; 4 9 6 -1; 11 -1 -1 -1; 7 6 7 -1; 6 -1 -1 -1; 5 9 7 -1]
 
  #--con_ary = [2 5 -1; 1 3 5; 2 4 -1; 3 5 -1; 1 2 4]
  #--e_weight = [4 4 -1; 4 4 4; 4 4 -1; 4 4 -1; 4 4 4]
  
  #### !!nodes_ary = [2 5 -1; 1 3 9; 2 7 -1; 5 8 -1; 1 4 6; 5 7 -1; 3 6 8; 4 7 9; 2 8 -1;;; 4 4 -1; 4 4 4; 4 4 -1; 4 4 -1; 4 4 4; 4 4 -1; 4 4 4; 4 4 4; 4 4 -1]
  #nodes_ary = [2 5 3; 1 9 -1; 7 1 -1; 5 8 -1; 1 4 6; 5 7 -1; 3 6 8; 4 7 9; 2 8 -1;;; 4 4 4; 4 4 -1; 4 4 -1; 4 4 -1; 4 4 4; 4 4 -1; 4 4 4; 4 4 4; 4 4 -1]
  #nodes_ary = [5 13 14 -1 -1 -1 -1 -1 -1; 3 4 7 12 15 18 -1 -1 -1; 2 7 11 -1 -1 -1 -1 -1 -1; 2 7 10 14 15 -1 -1 -1 -1; 1 7 8 11 12 17 20 -1 -1; 8 12 13 -1 -1 -1 -1 -1 -1; 2 3 4 5 10 16 17 18 -1; 5 6 12 -1 -1 -1 -1 -1 -1; 11 14 -1 -1 -1 -1 -1 -1 -1; 4 7 19 -1 -1 -1 -1 -1 -1; 3 5 9 12 15 17 18 -1 -1; 2 5 6 8 11 14 17 -1 -1; 1 6 19 20 -1 -1 -1 -1 -1; 1 4 9 12 15 16 -1 -1 -1; 2 4 11 14 16 -1 -1 -1 -1; 7 14 15 20 -1 -1 -1 -1 -1; 5 7 11 12 18 -1 -1 -1 -1; 2 7 11 17 -1 -1 -1 -1 -1; 10 13 20 -1 -1 -1 -1 -1 -1; 5 13 16 19 -1 -1 -1 -1 -1;;; 9 4 13 -1 -1 -1 -1 -1 -1; 2 4 3 3 8 2 -1 -1 -1; 2 4 9 -1 -1 -1 -1 -1 -1; 4 3 11 5 9 -1 -1 -1 -1; 9 9 5 4 6 3 2 -1 -1; 4 5 2 -1 -1 -1 -1 -1 -1; 3 4 3 9 4 3 6 3 -1; 5 4 4 -1 -1 -1 -1 -1 -1; 8 9 -1 -1 -1 -1 -1 -1 -1; 11 4 4 -1 -1 -1 -1 -1 -1; 9 4 8 8 8 8 11 -1 -1; 3 6 5 4 8 8 5 -1 -1; 4 2 4 3 -1 -1 -1 -1 -1; 13 5 9 8 5 9 -1 -1 -1; 8 9 8 5 8 -1 -1 -1 -1; 3 9 8 2 -1 -1 -1 -1 -1; 3 6 8 5 6 -1 -1 -1 -1; 2 3 11 6 -1 -1 -1 -1 -1; 4 4 9 -1 -1 -1 -1 -1 -1; 2 3 2 9 -1 -1 -1 -1 -1]
  
  # edges_ary = get_edges_ary(con_ary)
  global N = 8 #maximum(con_ary)
  global S = 20
  global E = 0#maximum(edges_ary)

  # weights = zeros(Float32, S)
  # nodes_list = matrixToAdjList(con_ary)
  # esize = esizeFromNodesAry(con_ary, e_weight)
  # _features = cat( esize, weights', dims=1)
  # sparse_adj_m = adjListToSparseAdjMatrix(nodes_list)
  nf = fill(0.0f0, 1, N) 

  n = 5
  p = 0.8
  graph = LightGraphs.SimpleGraphs.erdos_renyi(N, S)
  n_ed = LightGraphs.ne(graph)
  A = SparseArray(LightGraphs.adjacency_matrix(graph))
  #A = sparse([2, 3, 4, 5, 7, 8, 9, 1, 3, 4, 6, 7, 8, 9, 1, 2, 4, 5, 6, 7, 8, 1, 2, 3, 6, 1, 3, 7, 8, 2, 3, 4, 7, 8, 9, 1, 2, 3, 5, 6, 8, 9, 1, 2, 3, 5, 6, 7, 1, 2, 6, 7], [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 9, 9)
  #ef = [4 256; 4 256; 4 64; 4 256; 4 1024; 4 256]
  ef = fill(2.0f0, 2, n_ed)
  
  fg = FeaturedGraph(A, ef = ef, nf = nf)
  #print(nodes_list)
  #print(esize)
  #print(fg)
  #print(edges_ary)

  for edge in GraphSignals.edges(fg)
    (e, (n1, n2)) = edge
    indices = [incident_edges(fg.graph, n1); incident_edges(fg.graph, n2)]
    
    ef[2, e] = prod([ef[1, i] for i in indices])/ef[1, e]
  end
  
 
  fg.ef = ef
  

  
  
  amask = trues(S)
  history = Int[]
  
  return GameEnv(fg,  0.0, collect(1:S), amask, false, history)

end

#Don't know how to update action_mask, given a path

#function GI.set_path!(env::GameEnv, path)
#    env.history = nothing
#    env.path = path
#
#    update_actions_mask!(env)
#    any(env.amask) || (env.finished = true)
#  end
function GI.set_state!(env::GameEnv, state)
    #print("\n \n set state \n \n")
    env.fg = state.fg
    env.history = state.history
    env.o_edge = state.o_edge
    env.finished = state.finished
    env.reward = state.reward
    env.amask = state.amask
    return
end

function GI.clone(env::GameEnv)
    #print("\n \n CLONED \n \n")
    history = isnothing(env.history) ? nothing : copy(env.history)
    return GameEnv(copy(env.fg), copy(env.reward), copy(env.o_edge), copy(env.amask), copy(env.finished), history)
end

GI.two_players(::GameSpec) = false
 

GI.actions(::GameSpec) = collect(1:S)
GI.hasgraph(::GameSpec) = true


history(env::GameEnv) = env.history

#####
##### Defining game rules
#####


function update_actions_mask!(env::GameEnv, action)
    env.amask[action] = false
end

GI.actions_mask(env::GameEnv) = env.amask

# Update the game status
function update_status!(env::GameEnv, action)
    
    last_true_index = findlast(x -> x == true, env.amask)
    env.amask[last_true_index] = false
    env.finished = !any(env.amask)

end




  
#n0 = reduce(+, [node_dims_arys[i,:] for i in findall(n0_B)])
    
#n1 = reduce(+, [node_dims_arys[i,:] for i in findall(n1_B)])

function GI.play!(env::GameEnv, action)
  
  update_status!(env, action)
  flop = edge_feature(env.fg)[2, action]
  isnothing(env.history) || push!(env.history, env.o_edge[action])
  if env.finished == false
    ef = collect(edge_feature(env.fg))
    ef = [c[:] for c in eachcol(ef)]
    A = SparseArray(copy(env.fg.graph.S))
    
    

    
    edge = action #- sum(.!amask[1:action])
    n1i, n2i = get_nodes_from_edge(env.fg.graph, edge)

    
    
    A[n1i, n2i] = 0
    A[n2i, n1i] = 0
    

    delete =[edge]
    for i in enumerate(A[n1i,:])
      om = i[2], A[n2i, i[1]]
    
        if i[2]==1 && A[n2i, i[1]] ==1
            j = get_edge_from_nodes(env.fg.graph, i[1], n2i)
            k = get_edge_from_nodes(env.fg.graph, i[1], n1i)
            last_true_index = findlast(x -> x == true, env.amask)
            env.amask[last_true_index] = false
            push!(delete, j)
            ef[k][1] *= ef[j][1] 


          end
        A[n2i, i[1]] = (i[2] == 1)||(A[n2i, i[1]] == 1) ? 1 : 0
        A[i[1], n2i] = (i[2] == 1)||(A[i[1], n2i] == 1) ? 1 : 0
        A[n1i, i[1]] = 0
        A[i[1], n1i] = 0
    end

  

    A = A[setdiff(1:end, n1i), setdiff(1:end, n1i)]
    
    deleteat!(env.o_edge, sort(delete))
    deleteat!(ef, sort(delete))
    edg = [e for e in GraphSignals.edges(env.fg)][1:GraphSignals.ne(env.fg)]
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

    ef = ef[s]



    env.o_edge = env.o_edge[s]
    nf= fill(0.0f0, 1, size(A, 1))
    env.fg = FeaturedGraph(A, ef=hcat(ef...), nf=nf)
    edg = [e for e in GraphSignals.edges(env.fg)][1:GraphSignals.ne(env.fg)]
    update_e = incident_edges(env.fg.graph, n2i)
    for i in update_e
      (e, (n1, n2)) = edg[i]
      indices = [incident_edges(env.fg.graph, n1); incident_edges(env.fg.graph, n2)]
      ef[e][2] = prod([ef[i][1] for i in indices])/ef[e][1]
    end
    env.fg.ef = hcat(ef...)
    env.fg.nf = fill(0.0f0, 1, size(A, 1))
  end
  
  env.reward = -flop
  
  


end


  
    


  
GI.current_state(env::GameEnv) = (fg = copy(env.fg), reward = copy(env.reward), o_edge = copy(env.o_edge), amask = copy(env.amask), finished = copy(env.finished), history = copy(env.history))
  
GI.white_playing(env::GameEnv) = true




#####
##### Reward shaping
#####



function GI.game_terminated(env::GameEnv)
    #print(all(all(x -> x == true, vec) for vec in env.node_bool_arys))
    return env.finished
end

  
  
function GI.white_reward(env::GameEnv)
    #print("white_reward")
  R = env.reward
  
    return R
end




function GI.vectorize_state(::GameSpec, state)

  S = size(state.fg.graph.S, 1)
  S_new = max(S, N)
  
  S_matrix = Matrix(state.fg.graph.S)
  S_matrix_new = zeros(Float32, S_new, S_new)
  S_matrix_new[1:S, 1:S] = S_matrix
  return convert(Array{Float32}, S_matrix_new)
end

function GI.GetGraph(::GameSpec, state)
  return state.fg
end










GI.action_string(::GameSpec, a) = string(a)

function GI.parse_action(g::GameSpec, str)
    try
      p = parse(Int, str)
      1 <= p <= S ? p : nothing
    catch
      nothing
    end
  end

function GI.render(env::GameEnv, with_position_names=true, botmargin=true)
    
    print("\n GRAPH: \n")
    print(env.fg)
    print("\n AMASK: \n")
    print(env.amask)
    print("\n REWARD LIST: \n")
    print(env.reward)

    botmargin && print("\n")
  end
