from typing import Dict, Tuple
import re


from typing import Dict, Tuple

def calculate_per_core_performance(attributes: Dict, precision: str = 'fp64') -> Tuple[Dict, str]:
    unit_key = 'fp64_units' if precision == 'fp64' else 'fp32_units'
    
    base_clock = attributes.get('base_clock_ghz')
    simd = attributes.get('simd_lanes')
    units = attributes.get(unit_key)

    if base_clock is None:
        return None, f"Missing 'base_clock_ghz' to compute per-core {precision.upper()} performance."
    if simd is None:
        return None, f"Missing 'simd_lanes' to compute per-core {precision.upper()} performance."
    if units is None:
        return None, f"Missing '{unit_key}' to compute per-core {precision.upper()} performance."

    value = base_clock * simd * units
    return {
        'value': value,
        'formula': f"{base_clock} × {simd} × {units}"
    }, None

def generate_per_core_response(model_id: str, result: Dict, precision: str = 'fp64') -> str:
    label = "FP64" if precision == 'fp64' else "FP32"
    return f"""
🔹 Per-Core {label} Performance for {model_id}:
We use the following formula:
{label}_per_core = base_clock_ghz × simd_lanes × {label.lower()}_units

Extracted values:
  - base_clock_ghz = {result['formula'].split(' × ')[0]}
  - simd_lanes     = {result['formula'].split(' × ')[1]}
  - {label.lower()}_units   = {result['formula'].split(' × ')[2]}

Substituted:
{label}_per_core = {result['formula']} = {result['value']:.2f} GFLOPS
""".strip()
