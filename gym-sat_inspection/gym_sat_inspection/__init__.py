from gym.envs.registration import register

register(
    id='sat_inspection-v0',
    entry_point='gym_sat_inspection.envs:SatInspectionEnv',
)