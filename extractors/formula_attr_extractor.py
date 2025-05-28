import re
from typing import Dict, Union

def extract_processor_attributes(context: str, model_id: str) -> Dict[str, Union[str, float, int]]:
    attributes = {
        'socket_number': None,
        'cores': None,
        'base_clock_ghz': None,
        'simd_lanes': None,
        'fp64_units': None,
        'fp32_units': None
    }

    def get_attr(pattern: str) -> Union[float, None]:
        match = re.search(rf"{model_id} has_(?:attribute|parsed_attribute) {pattern}:([\d\.]+)", context)
        return float(match.group(1)) if match else None

    socket_match = re.search(rf"{model_id} has_(?:attribute|parsed_attribute|metadata) socket_number:([^\n]+)", context)
    if socket_match:
        attributes['socket_number'] = socket_match.group(1).strip().lower()

    attributes['cores'] = get_attr(r"(?:cores|number_of_cores)")
    attributes['base_clock_ghz'] = get_attr(r"base_clock_ghz")
    attributes['simd_lanes'] = get_attr(r"simd_units")
    attributes['fp64_units'] = get_attr(r"(?:fp_units|fp64_units)")
    attributes['fp32_units'] = get_attr(r"(?:fp32_units|fp_units)")

    # SIMD fallback using AVX detection if not directly provided
    if attributes['simd_lanes'] is None:
        if f"{model_id} has_feature AVX-512" in context:
            attributes['simd_lanes'] = 8
        elif f"{model_id} has_feature AVX2" in context:
            attributes['simd_lanes'] = 4
        elif f"{model_id} has_feature AVX" in context:
            attributes['simd_lanes'] = 2

    # ✅ FP32 fallback logic (only when missing but FP64 is present)
    if attributes['fp32_units'] is None and attributes['fp64_units'] is not None:
        attributes['fp32_units'] = 2 * attributes['fp64_units']

    return attributes

