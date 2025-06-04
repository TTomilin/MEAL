from .storm_2p import InTheGrid_2p

class LevelOne(InTheGrid_2p):
    def __init__(self, 
                 obs_size=5,
                 grid_size=8,
                 num_types=5,
                 num_coins=6,
                 num_coin_types=2,
                 num_inner_steps=152,
                 num_outer_steps=3,
                 fixed_coin_location=True,
                 num_agents=2,
                 payoff_matrix=None,
                 freeze_penalty=5):
        super().__init__(
            obs_size=obs_size,
            grid_size=grid_size,
            num_types=num_types,
            num_coins=num_coins,
            num_coin_types=num_coin_types,
            num_inner_steps=num_inner_steps,
            num_outer_steps=num_outer_steps,
            fixed_coin_location=fixed_coin_location,
            num_agents=num_agents
        )
