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
      num_games=100,
      num_workers=5,
      batch_size=5,
      use_gpu=true,
      reset_every=25,
      flip_probability=0.,
      alternate_colors=false),
    mcts=MctsParams(
      num_iters_per_turn=50,
      cpuct=1.0,
      temperature=PLSchedule([0, 20, 30], [1.0, 1.0, 0.03]),
      dirichlet_noise_ϵ=0.25,
      dirichlet_noise_α=0.03))
  
  arena = ArenaParams(
    sim=SimParams(
      num_games=100,
      num_workers=5,
      batch_size=5,
      use_gpu=true,
      reset_every=25,
      flip_probability=0,
      alternate_colors=false),
    mcts=MctsParams(
      self_play.mcts,
      temperature=ConstSchedule(0.2),
      dirichlet_noise_ϵ=0.05),
    update_threshold=0.05)
  
  learning = LearningParams(
    use_gpu=true,
    use_position_averaging=true,
    samples_weighing_policy=LOG_WEIGHT,
    batch_size=5,
    loss_computation_batch_size=5,
    optimiser=Adam(lr=2e-3),
    l2_regularization=1e-4,
    nonvalidity_penalty=1.,
    min_checkpoints_per_epoch=1,
    max_batches_per_checkpoint=20,
    num_checkpoints=5)
  
  params = Params(
    arena=arena,
    self_play=self_play,
    learning=learning,
    num_iters=14,
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
        num_iters_per_turn=50,
        cpuct=1.))
  
  minmax_baseline = Benchmark.MinMaxTS(
    depth=5,
    τ=0.2,
    amplify_rewards=true)
  
  alphazero_player = Benchmark.Full(arena.mcts)
  
  network_player = Benchmark.NetworkOnly(τ=0.5)
  
  benchmark_sim = SimParams(
    arena.sim;
    num_games=125,
    num_workers=5,
    batch_size=5,
    alternate_colors=false)
  
      
benchmark = [
  Benchmark.Single(
    Benchmark.Full(self_play.mcts),
    benchmark_sim),
    Benchmark.Single(
        Benchmark.NetworkOnly(),
        benchmark_sim)]
experiment = AlphaZero.Experiment("TensorContraction_N10_S20_PairNorm", GameSpec(), params, Network, netparams, benchmark)
