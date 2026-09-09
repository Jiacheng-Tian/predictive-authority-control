from __future__ import annotations

import dataclasses
import unittest

import numpy as np


class ActuatorModelTest(unittest.TestCase):
    def test_shared_actuator_applies_per_channel_slew_and_reset(self):
        from pac.simulation.actuators import ActuatorLimits, SharedActuator

        actuator = SharedActuator(ActuatorLimits(-1.0, 1.0, 0.1))
        first = actuator.apply(np.ones(6))
        second = actuator.apply(np.ones(6))
        reverse = actuator.apply(-np.ones(6))

        np.testing.assert_allclose(first.applied, 0.1)
        np.testing.assert_allclose(second.applied, 0.2)
        np.testing.assert_allclose(reverse.applied, 0.1)
        self.assertAlmostEqual(first.rate_limited_fraction, 1.0)
        self.assertAlmostEqual(first.amplitude_clipped_fraction, 0.0)

        actuator.reset()
        np.testing.assert_array_equal(actuator.apply(np.zeros(6)).applied, np.zeros(6))

    def test_amplitude_clip_precedes_slew_and_reports_fractions(self):
        from pac.simulation.actuators import ActuatorLimits, SharedActuator

        actuator = SharedActuator(ActuatorLimits(-0.5, 0.5, 0.2))
        step = actuator.apply(np.array([2.0, -2.0, 0.0, 0.0, 0.0, 0.0]))

        np.testing.assert_array_equal(step.requested, [2.0, -2.0, 0, 0, 0, 0])
        np.testing.assert_array_equal(step.amplitude_clipped, [0.5, -0.5, 0, 0, 0, 0])
        np.testing.assert_array_equal(step.applied, [0.2, -0.2, 0, 0, 0, 0])
        self.assertAlmostEqual(step.amplitude_clipped_fraction, 2.0 / 6.0)
        self.assertAlmostEqual(step.rate_limited_fraction, 2.0 / 6.0)

    def test_limits_and_requests_reject_invalid_values(self):
        from pac.simulation.actuators import ActuatorLimits, SharedActuator

        for kwargs in (
            dict(command_min=np.nan, command_max=1.0),
            dict(command_min=-1.0, command_max=np.inf),
            dict(command_min=10**1000, command_max=1.0),
            dict(command_min=1.0, command_max=-1.0),
            dict(command_min=2.0, command_max=3.0),
            dict(command_min=-3.0, command_max=-2.0),
            dict(command_min=-2.0, command_max=1.0),
            dict(command_min=-1.0, command_max=2.0),
            dict(command_min=-1.0, command_max=1.0, max_delta_per_step=np.nan),
            dict(command_min=-1.0, command_max=1.0, max_delta_per_step=-0.1),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ActuatorLimits(**kwargs)

        actuator = SharedActuator(ActuatorLimits(-1.0, 1.0))
        with self.assertRaises(ValueError):
            actuator.apply(np.zeros((6, 1)))
        with self.assertRaises(ValueError):
            actuator.apply(np.full(6, np.inf))

    def test_step_is_frozen_and_arrays_are_independent_copies(self):
        from pac.simulation.actuators import ActuatorLimits, ActuatorStep, SharedActuator

        requested = np.ones(6)
        step = SharedActuator(ActuatorLimits(-1.0, 1.0)).apply(requested)
        self.assertTrue(dataclasses.is_dataclass(step))
        self.assertIsInstance(step, ActuatorStep)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            step.applied = np.zeros(6)
        requested[0] = 9.0
        self.assertEqual(step.requested[0], 1.0)
        arrays = (step.requested, step.amplitude_clipped, step.applied)
        for array in arrays:
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array[0] = 0.0
        for index, left in enumerate(arrays):
            for right in arrays[index + 1:]:
                self.assertFalse(np.shares_memory(left, right))

    def test_simulator_metrics_and_forces_use_applied_action(self):
        from pac.simulation.core import AUVSimulator

        simulator = AUVSimulator(
            scenario=1,
            max_steps=2,
            actuator_max_delta_per_step=0.1,
        )
        simulator.reset(seed=0)
        requested = np.ones(6)
        _, info = simulator.step(requested)

        np.testing.assert_array_equal(info["requested_action"], requested)
        np.testing.assert_allclose(info["amplitude_clipped_action"], requested)
        np.testing.assert_allclose(info["applied_action"], 0.1)
        self.assertAlmostEqual(info["actuator_amplitude_clipped_fraction"], 0.0)
        self.assertAlmostEqual(info["actuator_rate_limited_fraction"], 1.0)
        self.assertAlmostEqual(info["energy"], 6.0 * 0.1**2)
        self.assertAlmostEqual(info["reward_smoothness"], 6.0 * 0.1**2)
        expected_wrench, expected_forces = simulator._action_to_wrench(np.full(6, 0.1))
        np.testing.assert_allclose(info["applied_wrench"], expected_wrench)
        np.testing.assert_allclose(info["thruster_forces"], expected_forces)

        simulator.reset(seed=0)
        _, reset_info = simulator.step(np.zeros(6))
        np.testing.assert_array_equal(reset_info["applied_action"], np.zeros(6))


if __name__ == "__main__":
    unittest.main()
