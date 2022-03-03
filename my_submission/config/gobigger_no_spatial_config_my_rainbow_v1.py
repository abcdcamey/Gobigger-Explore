from easydict import EasyDict

gobigger_config = dict(
    exp_name='gobigger_my_v1',
    env=dict(
        collector_env_num=2,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=1e10,
        team_num=4,
        player_num_per_team=3,
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
        type='rainbow',
        cuda=True,
        on_policy=False,
        priority=False, #defalut False
        priority_IS_weight=False,#defalut False
        nstep=6,
        discount_factor=0.95,
        model=dict(
            scalar_shape=5,
            food_map_shape=2,
            clone_with_food_relation_shape=150,
            clone_with_thorn_relation_shape=10,
            clones_shape=19,
            clone_with_team_relation_shape=12,
            clone_with_enemy_relation_shape=12,
            hidden_shape=128,
            encode_shape=32,
            action_type_shape=16,
            v_min=-10,
            v_max=10,
            n_atom=51,
        ),
        learn=dict(
            multi_gpu=False,
            update_per_collect=8,
            batch_size=56,
            learning_rate=0.001,
            target_theta=0.005,
            discount_factor=0.9,
            ignore_done=False,
            target_update_freq=100,
            learner=dict(
                hook=dict(save_ckpt_after_iter=1000, load_ckpt_before_run='gobigger_simple_baseline_dqn/ckpt/',)),
        ),
        collect=dict(n_sample=128, unroll_len=1, alpha=1.0),
        eval=dict(evaluator=dict(eval_freq=2000,)),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.5,
                decay=100000,
            ),
            replay_buffer=dict(
                replay_buffer_size=100000,
                alpha=0.6,
                beta=0.4,
                anneal_step=100000,
            ),
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