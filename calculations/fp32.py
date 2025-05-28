from typing import Dict, Tuple
import re

def calculate_fp32_performance(attributes: Dict) -> Tuple[Dict, str]:
    required = ['socket_number', 'cores', 'base_clock_ghz', 'simd_lanes', 'fp32_units']
    for key in required:
        if attributes.get(key) is None:
            return None, f"Missing '{key}' for FP32 performance calculation."

    socket_number_raw = attributes['socket_number'].lower()
    print('[DEBUG] socket_number:', socket_number_raw)

    # Parse possible socket values
    sockets_to_calculate = []
    if any(s in socket_number_raw for s in ["1p", "1s"]):
        sockets_to_calculate.append(1)
    if any(s in socket_number_raw for s in ["2p", "2s", "1p / 2p", "1s / 2s", "s2s"]):
        sockets_to_calculate.append(2)
    if not sockets_to_calculate:
        sockets_to_calculate.append(1)  # Fallback to 1 socket

    results = {}
    for sockets in sorted(set(sockets_to_calculate)):
        fp32 = (
            sockets *
            attributes['cores'] *
            attributes['base_clock_ghz'] *
            attributes['simd_lanes'] *
            attributes['fp32_units']
        )
        results[f"{sockets}-socket"] = {
            'value': fp32,
            'formula': f"{sockets} × {attributes['cores']} × {attributes['base_clock_ghz']} × "
                       f"{attributes['simd_lanes']} × {attributes['fp32_units']}"
        }

    return results, None


def generate_fp32_response(model_id: str, results: Dict) -> str:
    response = [f"FP32 Performance Breakdown for {model_id}:"]
    response.append("\nWe use the following formula to compute theoretical single precision performance:")
    response.append("FP32 GFLOPS = socket_number × cores × base_clock_ghz × simd_lanes × fp32_units")

    for config, data in results.items():
        response.append(f"\n🔹 {config.replace('-', ' ').title()} FP32: {data['value']:.2f} GFLOPS")
        response.append(f"   Formula: {data['formula']}")

    return "\n".join(response)
