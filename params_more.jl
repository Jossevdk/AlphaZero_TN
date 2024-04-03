Network = NetLib.SimpleNet

netparams = NetLib.SimpleNetHP(
  width=100,
  depth_common=10,
  use_batch_norm=false,
  batch_norm_momentum=1.)


  self_play = SelfPlayParams(
    sim=SimParams(
      num_games=128,
      num_workers=32,
      batch_size=16,
      use_gpu=true,
      reset_every=10,
      flip_probability=0.,
      alternate_colors=false),
    mcts=MctsParams(
      num_iters_per_turn=10,
      cpuct=1.0,
      temperature=ConstSchedule(0.),
      dirichlet_noise_ϵ=0.,
      dirichlet_noise_α=1.))
  
  arena = ArenaParams(
    sim=SimParams(
      num_games=100,
      num_workers=32,
      batch_size=16,
      use_gpu=false,
      reset_every=5,
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
    loss_computation_batch_size=1,
    optimiser=Adam(lr=2e-3),
    l2_regularization=1e-4,
    nonvalidity_penalty=1.,
    min_checkpoints_per_epoch=20,
    max_batches_per_checkpoint=2000,
    num_checkpoints=10)
  
  params = Params(
    arena=arena,
    self_play=self_play,
    learning=learning,
    num_iters=10,
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
    num_games=128,
    num_workers=32,
    batch_size=16,
    alternate_colors=false)
  
      
benchmark = [
  Benchmark.Single(
    Benchmark.Full(self_play.mcts),
    benchmark_sim),
    Benchmark.Single(
        Benchmark.NetworkOnly(),
        benchmark_sim)]

experiment = AlphaZero.Experiment("TensorContraction", GameSpec(), params, Network, netparams, benchmark)