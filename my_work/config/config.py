from easydict import EasyDict

gobigger_config = dict(
    exp_name='gobigger_my_v1',
    env=dict(
        collector_env_num=2,
        evaluator_env_num=2,
        n_evaluator_episode=2,
        stop_value=1e10,
        team_num=4,
        player_num_per_team=3,
        match_time=20*1,
        map_height=1000,
        map_width=1000,
        resize_height=160,
        resize_width=160,
        spatial=False,
        speed=False,
        all_vision=False,
        train=False,
        save_bin=True,
        save_path="./save_path/",
        save_video=True,
        save_quality='low',
        load_bin=True,
        load_bin_path='./save_path/1b323d00-a280-11ec-8629-acde48001122.pkl',
        load_bin_frame_num=50,#载入前n个action,直接执行
        jump_to_frame_file='./save_path/frame_50.pkl',
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
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
            action_type_shape=20,
        ),
        learn=dict(
            update_per_collect=8,
            batch_size=128,
            learning_rate=0.001,
            target_theta=0.005,
            discount_factor=0.9,
            ignore_done=False,
            learner=dict(
                hook=dict(save_ckpt_after_iter=2000, load_ckpt_before_run='gobigger_simple_baseline_dqn/ckpt/',)),
        ),
        collect=dict(n_sample=192, unroll_len=1, alpha=1.0),
        eval=dict(evaluator=dict(eval_freq=2000,)),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.5,
                decay=90000,
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