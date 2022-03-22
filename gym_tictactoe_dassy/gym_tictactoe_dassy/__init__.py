from gym.envs.registration import register

register(
    id='tictactoe-v0',
    entry_point='gym_tictactoe_dassy.envs:TictactoeEnv',
)