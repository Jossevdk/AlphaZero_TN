Network = NetLib.GraphNet



# Network = NetLib.SimpleNet

# netparams = NetLib.SimpleNetHP(
#   width=10,
#   depth_common=2,
#   use_batch_norm=false,
#   batch_norm_momentum=1.)

env_params = EnvParams(N = 25, S = 34, use_baseline = true, eval_mode=EVALMODE, use_robust = true, use_feas_act=true, n_best_result = 40)


netparams = NetLib.GraphNetHP(
  input_dim=env_params.use_robust ? 9 : 3,
  hidden_dim=25,
  num_blocks =5)


  self_play = SelfPlayParams(
    sim=SimParams(
      num_games=280,#240
      num_workers=4,
      batch_size=4,
      use_gpu=false,
      reset_every=4,
      flip_probability=0.,
      alternate_colors=false),
    mcts=MctsParams(
      num_iters_per_turn=300, #120
      cpuct=2.0,
      prior_temperature=1.0,
      temperature=PLSchedule([0, 20, 30], [1.0, 1.0, 0.3]),
      dirichlet_noise_ϵ=0.25,
      dirichlet_noise_α=1.0))
  
  arena = ArenaParams(
    sim=SimParams(
      num_games=32,
      num_workers=1,
      batch_size=1,
      use_gpu=false,
      reset_every=1,
      flip_probability=0,
      alternate_colors=false),
    mcts=MctsParams(
      self_play.mcts,
      cpuct=2.0,
      num_iters_per_turn=round(self_play.mcts.num_iters_per_turn/2),
    temperature=ConstSchedule(0.2),
    dirichlet_noise_ϵ=0.05),
  update_threshold=0.55) ##### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  learning = LearningParams(
    use_gpu=false,
    use_position_averaging=true,
    samples_weighing_policy=LOG_WEIGHT,
    batch_size=32,
    loss_computation_batch_size=32,
    optimiser=Adam(lr=2e-3),
    l2_regularization=1e-4,
    nonvalidity_penalty=1.,
    min_checkpoints_per_epoch=1,
    max_batches_per_checkpoint=200,
    num_checkpoints=3)
  
  params = Params(
    env_params=env_params,
    arena=arena,
    self_play=self_play,
    learning=learning,
    num_iters=iteration,
    ternary_outcome=false,
    use_symmetries=false,
    memory_analysis=nothing,
    mem_buffer_size=PLSchedule(
    [      0,        3, 6],
    [6_000, 6_000, 7_000]))
  
  #####
  ##### Evaluation benchmark
  #####
  

  
  alphazero_player = Benchmark.Full(arena.mcts)
  
  network_player = Benchmark.NetworkOnly(τ=0.5)
  
  benchmark_sim = SimParams(
    arena.sim;
    num_games=40,
    num_workers=8,
    batch_size=2,
    alternate_colors=false)
  
      
benchmark = [
  # Benchmark.Single(
  #   Benchmark.Full(self_play.mcts),
  #   benchmark_sim),
    Benchmark.Single(
        Benchmark.NetworkOnly(),
        benchmark_sim)]
