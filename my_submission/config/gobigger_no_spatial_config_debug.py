from easydict import EasyDict

gobigger_config = dict(
    exp_name='gobigger_baseline_v030_debug',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=1e10,
        team_num=2,
        player_num_per_team=2,
        match_time=60*10,
        map_height=1000,
        map_width=1000,
        resize_height=160,
        resize_width=160,
        spatial=False,
        speed = False,
        all_vision = False,
        train = False,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        cuda=False,
        on_policy=False,
        priority=False,
        priority_IS_weight=False,
        #nstep=3,
        model=dict(
            scalar_shape=5,
            food_shape=2,
            food_relation_shape=150,
            thorn_relation_shape=12,
            clone_shape=17,
            clone_relation_shape=12,
            hidden_shape=128,
            encode_shape=32,
            action_type_shape=16,
        ),
        learn=dict(
            update_per_collect=8,
            batch_size=128,
            learning_rate=0.001,
            target_theta=0.005,
            discount_factor=0.99,
            ignore_done=False,
            learner=dict(
                hook=dict(save_ckpt_after_iter=1000, load_ckpt_before_run='gobigger_simple_baseline_dqn/ckpt/',)),
        ),
        collect=dict(n_sample=256, unroll_len=1, alpha=1.0),
        eval=dict(evaluator=dict(eval_freq=2000,)),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.45,
                decay=100000,
            ),
            replay_buffer=dict(replay_buffer_size=90000, ),
        ),
    ),
)
main_config = EasyDict(gobigger_config)
gobigger_create_config = dict(
    env=dict(
        type='gobigger',
        import_names=['dizoo.gobigger.envs.gobigger_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
create_config = EasyDict(gobigger_create_config)