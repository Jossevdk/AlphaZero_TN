using AlphaZero
using StaticArrays
import AlphaZero.GI
using Base: @kwdef




include("functions.jl")

global S = 0


struct GameSpec <: GI.AbstractGameSpec end

"""
Setup as GameEnv.
In this environment the state is a adjecancy matrix.
"""
mutable struct GameEnv <: GI.AbstractGameEnv
    edges::Matrix{Int64}
    node_dims_arys::Vector{Vector{Int64}}
    node_bool_arys::Vector{Vector{Bool}}
    reward::Int64
    amask::BitVector
    finished::Bool
    history :: Union{Nothing, Vector{Int}}
    
end


GI.spec(::GameEnv) = GameSpec()

function GI.init(::GameSpec)
  nodes_ary = [2 8 9 10; 1 3 5 6; 2 7 10 -1; 5 6 -1 -1; 2 4 -1 -1; 2 4 8 -1; 3 -1 -1 -1; 1 6 10 -1; 1 -1 -1 -1; 1 3 8 -1;;; 9 7 6 5; 9 5 4 4; 5 11 9 -1; 9 9 -1 -1; 4 9 -1 -1; 4 9 6 -1; 11 -1 -1 -1; 7 6 7 -1; 6 -1 -1 -1; 5 9 7 -1]
    #nodes_ary = [2 5 -1; 1 3 5; 2 4 -1; 3 5 -1; 1 2 4;;; 4 4 -1; 4 4 4; 4 4 -1; 4 4 -1; 4 4 4]
    #### !!nodes_ary = [2 5 -1; 1 3 9; 2 7 -1; 5 8 -1; 1 4 6; 5 7 -1; 3 6 8; 4 7 9; 2 8 -1;;; 4 4 -1; 4 4 4; 4 4 -1; 4 4 -1; 4 4 4; 4 4 -1; 4 4 4; 4 4 4; 4 4 -1]
    #nodes_ary = [2 5 3; 1 9 -1; 7 1 -1; 5 8 -1; 1 4 6; 5 7 -1; 3 6 8; 4 7 9; 2 8 -1;;; 4 4 4; 4 4 -1; 4 4 -1; 4 4 -1; 4 4 4; 4 4 -1; 4 4 4; 4 4 4; 4 4 -1]
    #nodes_ary = [5 13 14 -1 -1 -1 -1 -1 -1; 3 4 7 12 15 18 -1 -1 -1; 2 7 11 -1 -1 -1 -1 -1 -1; 2 7 10 14 15 -1 -1 -1 -1; 1 7 8 11 12 17 20 -1 -1; 8 12 13 -1 -1 -1 -1 -1 -1; 2 3 4 5 10 16 17 18 -1; 5 6 12 -1 -1 -1 -1 -1 -1; 11 14 -1 -1 -1 -1 -1 -1 -1; 4 7 19 -1 -1 -1 -1 -1 -1; 3 5 9 12 15 17 18 -1 -1; 2 5 6 8 11 14 17 -1 -1; 1 6 19 20 -1 -1 -1 -1 -1; 1 4 9 12 15 16 -1 -1 -1; 2 4 11 14 16 -1 -1 -1 -1; 7 14 15 20 -1 -1 -1 -1 -1; 5 7 11 12 18 -1 -1 -1 -1; 2 7 11 17 -1 -1 -1 -1 -1; 10 13 20 -1 -1 -1 -1 -1 -1; 5 13 16 19 -1 -1 -1 -1 -1;;; 9 4 13 -1 -1 -1 -1 -1 -1; 2 4 3 3 8 2 -1 -1 -1; 2 4 9 -1 -1 -1 -1 -1 -1; 4 3 11 5 9 -1 -1 -1 -1; 9 9 5 4 6 3 2 -1 -1; 4 5 2 -1 -1 -1 -1 -1 -1; 3 4 3 9 4 3 6 3 -1; 5 4 4 -1 -1 -1 -1 -1 -1; 8 9 -1 -1 -1 -1 -1 -1 -1; 11 4 4 -1 -1 -1 -1 -1 -1; 9 4 8 8 8 8 11 -1 -1; 3 6 5 4 8 8 5 -1 -1; 4 2 4 3 -1 -1 -1 -1 -1; 13 5 9 8 5 9 -1 -1 -1; 8 9 8 5 8 -1 -1 -1 -1; 3 9 8 2 -1 -1 -1 -1 -1; 3 6 8 5 6 -1 -1 -1 -1; 2 3 11 6 -1 -1 -1 -1 -1; 4 4 9 -1 -1 -1 -1 -1 -1; 2 3 2 9 -1 -1 -1 -1 -1]
    edges = get_edges_ary(nodes_ary)
    global S = maximum(edges)
    node_bool_arys = get_node_bool_arys(nodes_ary)
    node_dims_arys = get_node_dims_arys(nodes_ary)  
    history = Int[]

    
    return GameEnv(edges, node_dims_arys, node_bool_arys,  0, trues(maximum(edges)), false, history)

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
    env.edges = state.edges
    env.history = state.history
    env.node_dims_arys= state.node_dims_arys
    env.node_bool_arys = state.node_bool_arys
    env.finished = state.finished
    env.reward = state.reward
    env.amask = state.amask
    return
end

function GI.clone(env::GameEnv)
    #print("\n \n CLONED \n \n")
    history = isnothing(env.history) ? nothing : copy(env.history)
    return GameEnv(copy(env.edges), copy(env.node_dims_arys), copy(env.node_bool_arys), copy(env.reward), copy(env.amask), copy(env.finished), history)
end

GI.two_players(::GameSpec) = false
 

GI.actions(::GameSpec) = collect(1:S)



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
    
    update_actions_mask!(env, action)
    env.finished = !any(env.amask)

    true
end
  
function GI.play!(env::GameEnv, action)
    isnothing(env.history) || push!(env.history, action)

    update_status!(env, action)
  

    node_i0, node_i1 = first.(Tuple.(findall( x -> x == action, env.edges)))

    n0 = env.node_dims_arys[node_i0]
    n1 = env.node_dims_arys[node_i1]

    n0_B = env.node_bool_arys[node_i0]
    n1_B = env.node_bool_arys[node_i1]

    if_diff = !(n0_B[node_i1])
    if if_diff
        
        contract_dims = n0 .+ n1
        contract_dims[n0.*n1 .!= 0 ] .=n0[n0.*n1 .!= 0 ] .* n1[n0.*n1 .!= 0 ]
        
        contract_bool = n0_B .| n1_B

        flop = prod(filter(x -> x != 0, n0[.!contract_bool]))*prod(filter(x -> x != 0, n1[.!contract_bool]))*prod(filter(x -> x != 0, n0[contract_bool]))
        
        
        contract_dims[contract_bool] .= 0  
        
        env.node_dims_arys[contract_bool] .= [contract_dims]
        env.node_bool_arys[contract_bool] .= [contract_bool]

        env.reward = -flop
    end
   
end


  
    


  
GI.current_state(env::GameEnv) = (node_dims_arys = copy(env.node_dims_arys), node_bool_arys = copy(env.node_bool_arys), edges = copy(env.edges), history = copy(env.history), finished = copy(env.finished), amask = copy(env.amask), reward = copy(env.reward))
  
GI.white_playing(env::GameEnv) = true




#####
##### Reward shaping
#####



function GI.game_terminated(env::GameEnv)
    #print(all(all(x -> x == true, vec) for vec in env.node_bool_arys))
    return all(all(x -> x == true, vec) for vec in env.node_bool_arys)
end

  
  
function GI.white_reward(env::GameEnv)
    #print("white_reward")
  R = env.reward
  
    return R
end


function GI.vectorize_state(::GameSpec, state)
  return convert(Array{Float32}, cat(stack(state.node_dims_arys, dims=1), stack(state.node_bool_arys, dims=1), dims =3))
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
    
    print("\n BOOL ARRAY: \n")
    print(env.node_bool_arys)
    print("\n DIM ARRAY: \n")
    print(env.node_dims_arys)
    print("\n AMASK: \n")
    print(env.amask)
    print("\n REWARD LIST: \n")
    print(env.reward)

    botmargin && print("\n")
  end