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
import Statistics
using EinExprs

function get_cartesian_index(sg::SparseGraph, eidx::Int)
  r = rowvals(sparse(sg))
  idx = findfirst(x -> x == eidx, collect(sg.edges))
  i = r[idx]
  j = 1
  while idx > length(SparseArrays.getcolptr(SparseMatrixCSC(sparse(sg)), 1:j))
    j += 1
  end
  return (i, j)
end

function edges_nodes(fg::FeaturedGraph)
  edgeslist = collect(fg.graph.edges)
  nodes = []
  for (i, e) in enumerate(edgeslist)
    push!(nodes, (e, get_cartesian_index(fg.graph, e)))
  end
  return sort(unique(nodes))
end

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

mutable struct tensor_environment
  A::Vector{SparseArray{Int64,2}}
  sizes::Vector{Matrix{Float32}}
  best_result::Vector{Union{Float32,Nothing}}
  greedy_result::Vector{Union{Float32,Nothing}}

  function tensor_environment(num_workers, N, S, eval_mode)

    if eval_mode
      new_A = [eval_A for i in 1:num_workers]
      sizes = [eval_sizes for i in 1:num_workers]
      new_best_result = [-1 for i in 1:num_workers]
    else
      new_A = [SparseArray(LightGraphs.adjacency_matrix(gen_graph(N, S))) for i in 1:num_workers]
      sizes = [rand([2.0f0, 3.0f0, 4.0f0, 5.0f0, 6.0f0], (1, S)) for i in 1:num_workers]
      new_best_result = [-1 for i in 1:num_workers]
    end

    greedy_result = []
    for w in 1:num_workers
      tensors = EinExpr{Symbol}[]
      list = [Symbol(Char(i)) for i in 97:200]
      list_of_tensors = [Symbol[] for i in 1:size(new_A[w], 1)]
      k = 1
      l = 1
      sizedict = Dict{Symbol,Float32}()
      for i in 1:size(new_A[w], 1)
        for j in i:size(new_A[w], 2)
          if new_A[w][i, j] == 1
            push!(list_of_tensors[i], list[k])
            push!(list_of_tensors[j], list[k])
            if j > i
              sizedict[list[k]] = sizes[w][1, l]
              l += 1
            end
            k += 1
          end
        end
      end
      for i in 1:size(new_A[w], 1)
        push!(tensors, EinExpr(list_of_tensors[i]))
      end

      expr = sum(tensors)
      path = SizedEinExpr(expr, sizedict)
      push!(greedy_result, einexpr(Greedy(metric=flops), path)[2])
    end
    new(new_A, sizes, new_best_result, greedy_result)
  end
end


#include("functions.jl")


global GREEDY_IND = 3
global best_result = 0

N = env_params.N
S = env_params.S






function get_robust_features(normalized_slice, median, clip_bound, fn)
  other_end = fn(normalized_slice)
  robust_scale = other_end .- median
  special_treatment_indices = isapprox.(robust_scale, 0)

  if any(special_treatment_indices)
    robust_scale[special_treatment_indices] .= Statistics.mean(robust_scale[special_treatment_indices])
    special_treatment_indices = isapprox.(robust_scale, 0)
    robust_scale[special_treatment_indices] .= 1.0
  end

  robust_feature = (normalized_slice .- median) ./ robust_scale
  robust_feature = clamp.(robust_feature, -clip_bound, clip_bound)
  return robust_feature
end




struct GameSpec <: GI.AbstractGameSpec end

"""
Setup as GameEnv.
In this environment the state is a adjecancy matrix.
"""
mutable struct GameEnv <: GI.AbstractGameEnv
  fg::AbstractFeaturedGraph
  attr::Matrix{Float32}
  total_reward::Float32
  reward::Float32
  o_edge::Vector{Int}
  amask::BitVector
  finished::Bool
  eval_mode::Bool
  history::Union{Nothing,Vector{Int}}
  greedy_result::Float32
end


GI.spec(::GameEnv) = GameSpec()


lock_ = ReentrantLock()

global self_play_tenv = tensor_environment(self_play.sim.num_workers, N, S, env_params.eval_mode)
global arena_tenv = [tensor_environment(arena.sim.num_workers, N, S, false) for i in 1:Int(ceil(arena.sim.num_games / arena.sim.num_workers))]
global benchmark_tenv = tensor_environment(benchmark[1].sim.num_workers, N, S, false)

function GI.init(::GameSpec; id::Union{Int64,Nothing}=0, worker_id::Union{Int64,Nothing}=1, reset_every::Union{Int64,Nothing}=nothing, network_name::Union{String,Nothing}=nothing)
  N = env_params.N
  S = env_params.S



  if isnothing(network_name)
    eval_mode = env_params.eval_mode
    global tenv = self_play_tenv

  elseif network_name == "self_play"
    eval_mode = false
    global tenv = self_play_tenv

  elseif network_name == "contender" || network_name == "baseline"
    eval_mode = true
    global tenv = arena_tenv[id+1]

    if network_name == "baseline" && id + 1 == arena.sim.num_games
      print("###### CHANGING TENV ARENA ######\n\n\n\n")
      global arena_tenv = [tensor_environment(arena.sim.num_workers, N, S, false) for i in 1:Int(ceil(arena.sim.num_games / arena.sim.num_workers))]
    end

  elseif network_name == "benchmark"
    eval_mode = true
    global tenv = benchmark_tenv
  end


  lock(lock_) do
    if !eval_mode && !isnothing(id) && !isnothing(reset_every)

      if id % reset_every == 0
        print("id: ", id, " worker_id: ", worker_id, " using new graph, ")
        tenv.A[worker_id] = SparseArray(LightGraphs.adjacency_matrix(gen_graph(N, S)))
        tenv.sizes[worker_id] = rand([2.0f0, 3.0f0, 4.0f0, 5.0f0, 6.0f0], (1, S))
        tenv.best_result[worker_id] = nothing

        if env_params.use_baseline
          tensors = EinExpr{Symbol}[]
          list = [Symbol(Char(i)) for i in 97:200]
          list_of_tensors = [Symbol[] for i in 1:size(tenv.A[worker_id], 1)]
          k = 1
          l = 1
          sizedict = Dict{Symbol,Float32}()
          for i in 1:size(tenv.A[worker_id], 1)
            for j in i:size(tenv.A[worker_id], 2)
              if tenv.A[worker_id][i, j] == 1
                push!(list_of_tensors[i], list[k])
                push!(list_of_tensors[j], list[k])
                if j > i
                  sizedict[list[k]] = tenv.sizes[worker_id][1, l]
                  l += 1
                end
                k += 1
              end
            end
          end
          for i in 1:size(tenv.A[worker_id], 1)
            push!(tensors, EinExpr(list_of_tensors[i]))
          end

          expr = sum(tensors)
          path = SizedEinExpr(expr, sizedict)

          tenv.best_result[worker_id] = -1
          tenv.greedy_result[worker_id] = einexpr(Greedy(metric=flops), path)[2]
        end
        print("new best result: ", tenv.best_result[worker_id], "\n", "id: ", id, "\n")
      end
    end


  end

  A = tenv.A[worker_id]


  nf = fill(0.0f0, 1, N)


  #A = SparseArray(sparse([2, 3, 6, 9, 1, 4, 1, 4, 7, 10, 2, 3, 7, 8, 9, 6, 8, 1, 5, 7, 10, 3, 4, 6, 8, 10, 4, 5, 7, 9, 10, 1, 4, 8, 10, 3, 6, 7, 8, 9], [1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 10, 10))

  ef = vcat(tenv.sizes[worker_id], fill(0.0f0, 8, S))
  attr = ef[1:3, :]

  fg = FeaturedGraph(A, ef=ef, nf=nf)

  for edge in edges_nodes(fg)
    (e, (n1, n2)) = edge
    indices1 = incident_edges(fg.graph, n1)
    indices2 = incident_edges(fg.graph, n2)
    indices = [indices1; indices2]

    attr[2, e] = prod([attr[1, i] for i in indices]) / attr[1, e]
    #greedy
    attr[3, e] = prod([attr[1, i] for i in indices1]) + prod([attr[1, i] for i in indices2]) - attr[2, e] / attr[1, e]
  end




  normalized_slice = copy(attr)
  normalized_slice[GREEDY_IND, isapprox.(normalized_slice[GREEDY_IND, :], 0)] .= 1
  normalized_slice[GREEDY_IND, :] .= sign.(normalized_slice[GREEDY_IND, :]) .* log.(abs.(normalized_slice[GREEDY_IND, :]))

  normalized_slice .-= minimum(normalized_slice, dims=2)
  median = Statistics.median(normalized_slice, dims=2)

  robust_features = vcat(
    get_robust_features(normalized_slice, median, 100, x -> maximum(x, dims=2)),
    get_robust_features(normalized_slice, median, 100, x -> minimum(x, dims=2)),
    log.(1 .+ normalized_slice)
  )

  fg.ef = robust_features



  amask = trues(env_params.S)
  history = Int[]

  return GameEnv(fg, attr, 0.0, 0.0, collect(1:env_params.S), amask, false, eval_mode, history, copy(tenv.greedy_result[worker_id]))

end

function GI.set_state!(env::GameEnv, state)
  env.fg = state.fg
  env.attr = state.attr
  env.eval_mode = state.eval_mode
  env.history = state.history
  env.o_edge = state.o_edge
  env.finished = state.finished
  env.total_reward = state.total_reward
  env.reward = state.reward
  env.amask = state.amask
  env.greedy_result = state.greedy_result

  return
end

function GI.clone(env::GameEnv)
  history = isnothing(env.history) ? nothing : copy(env.history)
  return GameEnv(copy(env.fg), copy(env.attr), copy(env.total_reward), copy(env.reward), copy(env.o_edge), copy(env.amask), copy(env.finished), copy(env.eval_mode), history, copy(env.greedy_result))
end

GI.two_players(::GameSpec) = false


GI.actions(::GameSpec) = collect(1:env_params.S)
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

function GI.play!(env::GameEnv, action; worker_id=1, playing=false)


  # print(copy(env.fg.graph.S), "\n")
  # print(env.o_edge, env.o_edge[action], "\n")
  # print([e for e in GraphSignals.edges(env.fg)][1:GraphSignals.ne(env.fg)], "\n\n")

  update_status!(env, action)
  # if playing
  #   print("maximum action:" , maximum(env.attr[2, :])/env.greedy_result, "\n")
  #   print("minimum action:" , minimum(env.attr[2, :])/env.greedy_result, "\n")

  # end
  flop = env.attr[2, action]
  isnothing(env.history) || push!(env.history, env.o_edge[action])
  if env.finished == false

    attr = [c[:] for c in eachcol(env.attr)]
    A = SparseArray(copy(env.fg.graph.S))




    edge = action
    n1i, n2i = get_nodes_from_edge(env.fg.graph, edge)



    A[n1i, n2i] = 0
    A[n2i, n1i] = 0


    delete = [edge]
    for i in enumerate(A[n1i, :])

      if i[2] == 1 && A[n2i, i[1]] == 1
        j = get_edge_from_nodes(env.fg.graph, i[1], n2i)
        k = get_edge_from_nodes(env.fg.graph, i[1], n1i)
        last_true_index = findlast(x -> x == true, env.amask)
        env.amask[last_true_index] = false
        push!(delete, j)
        attr[k][1] *= attr[j][1]


      end
      A[n2i, i[1]] = (i[2] == 1) || (A[n2i, i[1]] == 1) ? 1 : 0
      A[i[1], n2i] = (i[2] == 1) || (A[i[1], n2i] == 1) ? 1 : 0
      A[n1i, i[1]] = 0
      A[i[1], n1i] = 0
    end


    update = true
    if all(A[1:end, n2i] .== 0)
      update = false
      A = A[setdiff(1:end, [n1i, n2i]), setdiff(1:end, [n1i, n2i])]
    else
      A = A[setdiff(1:end, n1i), setdiff(1:end, n1i)]
    end


    deleteat!(env.o_edge, sort(delete))
    deleteat!(attr, sort(delete))
    edg = edges_nodes(env.fg)
    deleteat!(edg, sort(delete))
    new_edg = []
    for ed in edg
      (e, (n1, n2)) = ed
      if n1 == n1i
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

    attr = hcat(attr[s]...)



    env.o_edge = env.o_edge[s]
    nf = fill(0.0f0, 1, size(A, 1))
    env.fg = FeaturedGraph(A, ef=vcat(attr, attr, attr), nf=nf)
    edg = edges_nodes(env.fg)
    if update
      update_e = incident_edges(env.fg.graph, n2i)
      for i in update_e
        (e, (n1, n2)) = edg[i]
        indices1 = incident_edges(env.fg.graph, n1)
        indices2 = incident_edges(env.fg.graph, n2)
        indices = [indices1; indices2]

        attr[2, e] = prod([attr[1, i] for i in indices]) / attr[1, e]
        #greedy
        attr[3, e] = prod([attr[1, i] for i in indices1]) + prod([attr[1, i] for i in indices2]) - attr[2, e] / attr[1, e]



      end
    end
    normalized_slice = copy(attr)

    normalized_slice[GREEDY_IND, isapprox.(normalized_slice[GREEDY_IND, :], 0)] .= 1
    normalized_slice[GREEDY_IND, :] .= sign.(normalized_slice[GREEDY_IND, :]) .* log.(abs.(normalized_slice[GREEDY_IND, :]))
    normalized_slice .-= minimum(normalized_slice, dims=2)
    median = Statistics.median(normalized_slice, dims=2)

    robust_features = vcat(
      get_robust_features(normalized_slice, median, 100, x -> maximum(x, dims=2)),
      get_robust_features(normalized_slice, median, 100, x -> minimum(x, dims=2)),
      log.(1 .+ normalized_slice)
    )

    env.fg.ef = robust_features

    env.fg.nf = fill(0.0f0, 1, size(A, 1))
    env.attr = attr


  end


  #env.reward = -flop / env.greedy_result
  env.total_reward += -flop / env.greedy_result

  penalty = 0
  # if playing
  #   print("total_reward: ", env.total_reward, "\n")
  # end
  #Block decreasing actions with estimated penalty
  lock(lock_) do
    # if playing
    #   print(env.eval_mode, env_params.use_feas_act, env.finished, "\n")
    # end
    if env_params.use_feas_act && env.finished
      # if playing
      #   print("total_reward in lock : ", env.total_reward, "\n")
      # end
      if isnothing(tenv.best_result[worker_id])
        tenv.best_result[worker_id] = env.total_reward
        if playing
          print("nothing \n \n")
        end  
      end
      if env.total_reward > tenv.best_result[worker_id]
        if playing
          print(playing, "finished  better!", worker_id, "\n\n")
        end
        tenv.best_result[worker_id] = env.total_reward
        env.reward = 1
      else
        if playing
          print("worse.... \n\n")
        end
        env.total_reward = tenv.best_result[worker_id]
      end
    end
  end

  lock(lock_) do
    if env_params.use_feas_act && env.finished == false
      if !isnothing(tenv.best_result[worker_id])
        if env.total_reward - minimum(attr[2, :]) / env.greedy_result < tenv.best_result[worker_id]
          if playing
            print("Prematurely finished \n")
            print("total_reward: ", env.total_reward- minimum(attr[2, :])/env.greedy_result, "\n")
          end

          env.finished = true
          env.reward = -1
          alpha = 0.5
          mean_best_result_cost = tenv.best_result[worker_id] / env_params.N
          mean_current_path_cost = env.total_reward / (env_params.N - size(A, 1) + 1e-8)
          future_term = 2 * (size(A, 1) - 1) * (mean_best_result_cost * alpha + mean_current_path_cost * (1 - alpha))
          penalty = abs(minimum(attr[2, :])) / env.greedy_result - future_term
          env.total_reward = tenv.best_result[worker_id]
          env.history = Int[]

          # if playing
          #   print("penalty: ", penalty, "\n")
          # end



          # if playing
          #   print("penalty: ", penalty, "\n")
          # end

        end
      end
      #env.reward -= penalty
      #env.total_reward -= penalty
    end
  end


end







function GI.current_state(env::GameEnv)
  history_copy = env.history === nothing ? nothing : copy(env.history)
  return (
    fg=copy(env.fg),
    attr=copy(env.attr),
    total_reward=copy(env.total_reward),
    reward=copy(env.reward),
    o_edge=copy(env.o_edge),
    amask=copy(env.amask),
    finished=copy(env.finished),
    eval_mode=copy(env.eval_mode),
    history=history_copy,
    greedy_result=copy(env.greedy_result)
  )
end
GI.white_playing(env::GameEnv) = true




#####
##### Reward shaping
#####



function GI.game_terminated(env::GameEnv)
  return env.finished
end



function GI.white_reward(env::GameEnv)
  R = env.reward

  return R
end




function GI.vectorize_state(::GameSpec, state)

  S = size(state.fg.graph.S, 1)
  S_new = max(S, env_params.N)

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
    1 <= p <= env_params.S ? p : nothing
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
