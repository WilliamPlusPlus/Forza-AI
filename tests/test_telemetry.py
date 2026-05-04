from __future__ import annotations

import struct
import unittest

from forza_ai.telemetry import (
    DASH_SHARED_SCHEMA,
    HORIZON_HEADER_SCHEMA,
    MOTORSPORT_DASH_SCHEMA,
    SLED_SCHEMA,
    ForzaPacketParser,
    _format,
    is_driving_frame,
    normalized_control_value,
)


def _pack(schema, overrides):
    values = []
    for name, fmt in schema:
        value = overrides.get(name, 0)
        if fmt == "f":
            value = float(value)
        values.append(value)
    return struct.pack(_format(schema), *values)


class ForzaPacketParserTests(unittest.TestCase):
    def test_motorsport_2023_dash_packet_is_331_bytes(self):
        packet = _pack(
            SLED_SCHEMA,
            {"is_race_on": 1, "timestamp_ms": 42, "current_engine_rpm": 6200.0},
        ) + _pack(
            MOTORSPORT_DASH_SCHEMA,
            {"speed": 71.5, "accel": 188, "brake": 3, "steer": -17, "track_ordinal": 812},
        )

        self.assertEqual(len(packet), 331)
        frame = ForzaPacketParser("motorsport_2023_dash").parse(packet)

        self.assertEqual(frame.values["is_race_on"], 1)
        self.assertAlmostEqual(frame.values["speed"], 71.5)
        self.assertEqual(frame.values["accel"], 188)
        self.assertAlmostEqual(normalized_control_value(frame, "accel"), 188 / 255.0)
        self.assertAlmostEqual(normalized_control_value(frame, "brake"), 3 / 255.0)
        self.assertEqual(frame.values["steer"], -17)
        self.assertEqual(frame.values["track_ordinal"], 812)

    def test_horizon_dash_packet_is_supported_as_primary_profile(self):
        packet = _pack(SLED_SCHEMA, {"is_race_on": 1}) + _pack(
            HORIZON_HEADER_SCHEMA,
            {"horizon_car_category": 11},
        ) + _pack(
            DASH_SHARED_SCHEMA,
            {"speed": 33.25, "accel": 122, "steer": 12},
        )

        self.assertEqual(len(packet), 323)
        frame = ForzaPacketParser("horizon_dash").parse(packet)

        self.assertEqual(frame.values["horizon_car_category"], 11)
        self.assertAlmostEqual(frame.values["speed"], 33.25)
        self.assertEqual(frame.values["steer"], 12)
        self.assertTrue(is_driving_frame(frame))

    def test_horizon_free_roam_can_be_drivable_when_race_flag_is_off(self):
        packet = _pack(SLED_SCHEMA, {"is_race_on": 0}) + _pack(
            HORIZON_HEADER_SCHEMA,
            {"horizon_car_category": 11},
        ) + _pack(
            DASH_SHARED_SCHEMA,
            {"speed": 0.0, "accel": 90, "brake": 0, "steer": 0},
        )

        frame = ForzaPacketParser("horizon_dash").parse(packet)

        self.assertEqual(frame.values["is_race_on"], 0)
        self.assertTrue(is_driving_frame(frame))


if __name__ == "__main__":
    unittest.main()
