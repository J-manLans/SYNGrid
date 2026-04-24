from syn_grid.gymnasium.observation_space.difficulty.base_difficulty import (
    BaseDifficulty,
)


class EasyDifficulty(BaseDifficulty):

    def get_observation(self) -> dict[str, NDArray]:
        agent_row, agent_col = self._world.DROID.position

        # NOTE: change here
        # ---- Agent ---- #
        self._agent_data[0] = agent_row
        self._agent_data[1] = agent_col

        # NOTE: change here
        # ---- Orbs ---- #
        active = self._world.get_orb_is_active_status(False)
        positions = self._world.get_orb_positions(False)
        remaining = self._world.get_orb_life()
        categories = self._world.get_orb_categories()
        types = self._world.get_orb_types()
        tiers = self._world.get_orb_tiers()

        for i in range(len(self._world.ALL_ORBS)):
            if active[i]:
                # NOTE: change here
                pos = positions[i]
                r_timer = remaining[i]
                r_cat = int(categories[i])
                r_type = int(types[i])
                r_tier = tiers[i]

                # NOTE: change here
                self._orb_data[i] = [
                    pos[0],
                    pos[1],
                    r_cat,
                    r_type,
                    r_tier,
                ]
            else:
                # NOTE: change here
                self._orb_data[i] = [-1, -1, -1, -1, -1, -1]

        return {"agent data": self._agent_data, "orbs data": self._orb_data}

    ...
