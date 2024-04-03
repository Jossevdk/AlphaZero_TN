Network = NetLib.GraphNet

netparams = NetLib.GraphNetHP(
  hidden_dim=2,
  output_dim =1,
  use_batch_norm=false,
  batch_norm_momentum=1.)

# Network = NetLib.SimpleNet

# netparams = NetLib.SimpleNetHP(
#   width=10,
#   depth_common=2,
#   use_batch_norm=false,
#   batch_norm_momentum=1.)


  self_play = SelfPlayParams(
    sim=SimParams(
      num_games=10,
      num_workers=1,
      batch_size=1,
      use_gpu=true,
      reset_every=2,
      flip_probability=0.,
      alternate_colors=false),
    mcts=MctsParams(
      num_iters_per_turn=3,
      cpuct=1.0,
      temperature=ConstSchedule(0.),
      dirichlet_noise_ϵ=0.,
      dirichlet_noise_α=1.))
  
  arena = ArenaParams(
    sim=SimParams(
      num_games=4,
      num_workers=1,
      batch_size=1,
      use_gpu=false,
      reset_every=1,
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
    batch_size=1,
    loss_computation_batch_size=1,
    optimiser=Adam(lr=2e-3),
    l2_regularization=1e-4,
    nonvalidity_penalty=1.,
    min_checkpoints_per_epoch=1,
    max_batches_per_checkpoint=2000,
    num_checkpoints=2)
  
  params = Params(
    arena=arena,
    self_play=self_play,
    learning=learning,
    num_iters=1,
    ternary_outcome=false,
    use_symmetries=false,
    memory_analysis=nothing,
    mem_buffer_size=PLSchedule(
    [      0,        15],
    [400_000, 1_000_000]))
  
  #####
  ##### Evaluation benchmark
  #####
  
  mcts_baseline =
    Benchmark.MctsRollouts(
      MctsParams(
        arena.mcts,
        num_iters_per_turn=10,
        cpuct=1.))
  
  minmax_baseline = Benchmark.MinMaxTS(
    depth=5,
    τ=0.2,
    amplify_rewards=true)
  
  alphazero_player = Benchmark.Full(arena.mcts)
  
  network_player = Benchmark.NetworkOnly(τ=0.5)
  
  benchmark_sim = SimParams(
    arena.sim;
    num_games=2,
    num_workers=1,
    batch_size=1,
    alternate_colors=false)
  
      
benchmark = [
  Benchmark.Single(
    Benchmark.Full(self_play.mcts),
    benchmark_sim),
    Benchmark.Single(
        Benchmark.NetworkOnly(),
        benchmark_sim)]

experiment = AlphaZero.Experiment("TensorContraction", GameSpec(), params, Network, netparams, benchmark)
