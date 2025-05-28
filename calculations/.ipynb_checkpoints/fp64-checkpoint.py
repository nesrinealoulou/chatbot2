import re
from typing import Dict, Tuple, Union


def calculate_fp64_performance(attributes: Dict) -> Tuple[Dict, str]:
    required = ['socket_number', 'cores', 'base_clock_ghz', 'simd_lanes', 'fp64_units']

    if any(attributes[k] is None for k in required):
        missing = [k for k in required if attributes[k] is None]
        error_msg = f"Missing required fields for FP64 performance calculation: {missing}"
        print(f"[ERROR] {error_msg}")
        return None, error_msg

    socket_number = attributes['socket_number']
    
    socket_number_lower = socket_number.lower()
    
    sockets_to_calculate = []

    match_1socket = re.search(r"\b1[p|s]\b|1p|1s", socket_number_lower)
    match_2socket = re.search(r"\b2[p|s]\b|2p|2s|s2s", socket_number_lower)


    if match_1socket:
        sockets_to_calculate.append(1)
    if match_2socket:
        sockets_to_calculate.append(2)
    if not sockets_to_calculate:
        print("[WARNING] No socket matches found, defaulting to 1 socket.")

    results = {}
    for sockets in sockets_to_calculate:
        fp64 = (sockets *
                attributes['cores'] *
                attributes['base_clock_ghz'] *
                attributes['simd_lanes'] *
                attributes['fp64_units'])
        formula = f"{sockets} × {attributes['cores']} × {attributes['base_clock_ghz']} × {attributes['simd_lanes']} × {attributes['fp64_units']}"
        results[f"{sockets}-socket"] = {'value': fp64, 'formula': formula}

    return results, None


def generate_fp64_response(model_id: str, results: Dict) -> str:
    response = [f"FP64 Performance Breakdown for {model_id}:"]
    response.append("\nWe use the following formula to compute theoretical double precision performance:")
    response.append("FP64 GFLOPS = socket_number × cores × base_clock_ghz × simd_lanes × fp64_units")
    for config, data in results.items():
        response.append(f"\n🔹 {config.replace('-', ' ').title()} FP64: {data['value']:.2f} GFLOPS")
        response.append(f"   Formula: {data['formula']}")
    return "\n".join(response)