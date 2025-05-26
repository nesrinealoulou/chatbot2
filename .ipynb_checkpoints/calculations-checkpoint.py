import re
from typing import Dict, Tuple, Union


def extract_processor_attributes(context: str, model_id: str) -> Dict[str, Union[str, float, int]]:
    """
    Extracts all relevant attributes for performance calculations from context.
    Handles socket formats like '1P', '2P', '1P / 2P', '1S only', 'S2S', etc.
    Returns a dictionary with extracted values or None if missing.
    """
    attributes = {
        'socket_config': None,
        'cores': None,
        'base_clock_ghz': None,
        'simd_lanes': None,
        'fp64_units': None
    }
    # print('context',context)

    # Helper function to extract numerical attribute values from the context
    def get_attr(pattern: str) -> Union[float, None]:
        match = re.search(rf"{model_id} has_(?:attribute|parsed_attribute) {pattern}:([\d\.]+)", context)
        return float(match.group(1)) if match else None

    # Robust extraction of socket configuration (graph-based or raw JSON fallback)
    # Robust extraction of socket configuration
    socket_match = re.search(rf"{model_id} has_(?:attribute|parsed_attribute|metadata) socket_number:([^\n]+)", context)
    if socket_match:
        attributes['socket_config'] = socket_match.group(1).strip().lower()
    else:
        # Fallback: raw JSON-style
        json_match = re.search(r'"socket_number"\s*:\s*"([^"]+)"', context, re.IGNORECASE)
        if json_match:
            attributes['socket_config'] = json_match.group(1).strip().lower()



    # Extract numerical specs
    attributes['cores'] = get_attr(r"(?:cores|number_of_cores)")
    attributes['base_clock_ghz'] = get_attr(r"base_clock_ghz")
    attributes['simd_lanes'] = get_attr(r"simd_units")
    attributes['fp64_units'] = get_attr(r"(?:fp_units|fp64_units)")

    # Infer SIMD lanes from features if not explicitly defined
    if attributes['simd_lanes'] is None:
        if f"{model_id} has_feature AVX-512" in context:
            attributes['simd_lanes'] = 8
        elif f"{model_id} has_feature AVX2" in context:
            attributes['simd_lanes'] = 4
        elif f"{model_id} has_feature AVX" in context:
            attributes['simd_lanes'] = 2

    return attributes


def calculate_fp64_performance(attributes: Dict) -> Tuple[Dict, str]:
    """
    Calculates FP64 performance based on extracted attributes
    Returns (results_dict, error_message)
    """
    # Validate required attributes
    missing = [k for k, v in attributes.items() if v is None]
    if missing:
        return None, f"Missing {', '.join(missing)}. Please provide to compute FP64 performance."
    
    # Determine socket configurations to calculate
    socket_config = attributes['socket_config']
    sockets_to_calculate = []
    
    # Match flexible patterns like "1p", "1s only", "1p / 2p", "s2s", etc.
    if re.search(r"\b1[p|s]\b", socket_config) or "1p only" in socket_config or "1s only" in socket_config or "1p" in socket_config:
        sockets_to_calculate.append(1)
    if re.search(r"\b2[p|s]\b", socket_config) or "2p only" in socket_config or "2s only" in socket_config or "2p" in socket_config or "s2s" in socket_config:
        sockets_to_calculate.append(2)
    
    # Fallback if nothing matched
    if not sockets_to_calculate:
        sockets_to_calculate.append(1)

    
    # Perform calculations
    results = {}
    for sockets in sockets_to_calculate:
        fp64 = (sockets * 
                attributes['cores'] * 
                attributes['base_clock_ghz'] * 
                attributes['simd_lanes'] * 
                attributes['fp64_units'])
        
        results[f"{sockets}-socket"] = {
            'value': fp64,
            'formula': f"{sockets} × {attributes['cores']} × {attributes['base_clock_ghz']} × {attributes['simd_lanes']} × {attributes['fp64_units']}"
        }
    
    # # Calculate per-core performance if needed
    # per_core = (attributes['base_clock_ghz'] * 
    #             attributes['simd_lanes'] * 
    #             attributes['fp64_units'])
    # results['per_core'] = {
    #     'value': per_core,
    #     'formula': f"{attributes['base_clock_ghz']} × {attributes['simd_lanes']} × {attributes['fp64_units']}"
    # }
    
    return results, None

def generate_fp64_response(model_id: str, results: Dict) -> str:
    """
    Generates a human-readable response from calculation results
    """
    response = [f"FP64 Performance Calculation for {model_id}:"]
    
    for config, data in results.items():
        if config == 'per_core':
            response.append(f"\n🔹 Per-core FP64: {data['value']:.2f} GFLOPS")
            response.append(f"   Formula: {data['formula']}")
        else:
            response.append(f"\n🔹 {config.replace('-', ' ').title()} FP64: {data['value']:.2f} GFLOPS")
            response.append(f"   Formula: {data['formula']}")
    
    return "\n".join(response)