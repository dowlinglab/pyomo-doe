"""Declared Tier 0 checks; separate from the parent study's sensitivity driver."""
CASES = {
    "right_hold": {"primary_hold": "right"},
    "initial_minus": {"initial_temperature_offset_C": -.165},
    "initial_plus": {"initial_temperature_offset_C": .165},
    "preload_one_second": {"preload_seconds": 1.},
    "ambient_four_degrees": {"ambient_offset_bounds_C": [-4., 4.]},
    "protocol_30s": {"record_filter": "sin_50_50"},
    "protocol_120s": {"record_filter": "sin_2_50"},
    "protocol_300s": {"record_filter": "sin_5_50"},
    "protocol_step": {"record_filter": "step_50"}
}
