from gym.envs.registration import register

"""Top-level package for Constraint Water Treatment Gym."""

__author__ = """Stefan Heid"""
__email__ = 'stefan.heid@upb.de'

register(
    id='distillation-plant-v0',
    entry_point='constraint_water_treatment_gym.envs:WaterTreatmentEnv',
)
