Network = NetLib.GraphNet



# Network = NetLib.SimpleNet

# netparams = NetLib.SimpleNetHP(
#   width=10,
#   depth_common=2,
#   use_batch_norm=false,
#   batch_norm_momentum=1.)

env_params = EnvParams(N = 25, S = nedges, use_baseline = true, eval_mode=EVALMODE, use_robust = true, use_feas_act=true, n_best_result = 40)


netparams = NetLib.GraphNetHP(
  input_dim=env_params.use_robust ? 9 : 3,
  hidden_dim=25,
  num_blocks =5)


  self_play = SelfPlayParams(
    sim=SimParams(
      num_games=500,
      num_workers=16,
      batch_size=8,
      use_gpu=false,
      reset_every=4,
      flip_probability=0.,
      alternate_colors=false),
    mcts=MctsParams(
      num_iters_per_turn=200,
      cpuct=2.0,
      prior_temperature=1.0,
      temperature=PLSchedule([0, 20, 30], [1.0, 1.0, 0.3]),
      dirichlet_noise_ϵ=0.25,
      dirichlet_noise_α=1.0))
  
  arena = ArenaParams(
    sim=SimParams(
      num_games=16,
      num_workers=16,
      batch_size=16,
      use_gpu=false,
      reset_every=2,
      flip_probability=0,
      alternate_colors=false),
    mcts=MctsParams(
      self_play.mcts,
    temperature=ConstSchedule(0.2),
    dirichlet_noise_ϵ=0.05),
  update_threshold=0.05)
  
  learning = LearningParams(
    use_gpu=false,
    use_position_averaging=true,
    samples_weighing_policy=LOG_WEIGHT,
    batch_size=128,
    loss_computation_batch_size=128,
    optimiser=Adam(lr=2e-3),
    l2_regularization=1e-4,
    nonvalidity_penalty=1.,
    min_checkpoints_per_epoch=1,
    max_batches_per_checkpoint=200,
    num_checkpoints=4)
  
  params = Params(
    env_params=env_params,
    arena=arena,
    self_play=self_play,
    learning=learning,
    num_iters=12,
    ternary_outcome=false,
    use_symmetries=false,
    memory_analysis=nothing,
    mem_buffer_size=PLSchedule(
    [      0,        3, 6],
    [10_000, 20_000, 30_000]))
  
  #####
  ##### Evaluation benchmark
  #####
  

  
  alphazero_player = Benchmark.Full(arena.mcts)
  
  network_player = Benchmark.NetworkOnly(τ=0.5)
  
  benchmark_sim = SimParams(
    arena.sim;
    num_games=40,
    num_workers=16,
    batch_size=2,
    alternate_colors=false)
  
      
benchmark = [
  # Benchmark.Single(
  #   Benchmark.Full(self_play.mcts),
  #   benchmark_sim),
    Benchmark.Single(
        Benchmark.NetworkOnly(),
        benchmark_sim)]
