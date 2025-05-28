import re
from typing import Dict, Tuple, Union


def extract_memory_attributes(context: str, model_id: str) -> Dict[str, Union[float, str]]:
    import re

    def extract_value(pattern):
        match = re.search(rf"{model_id} has_(?:attribute|parsed_attribute) {pattern}:([\d\.]+)", context)
        return float(match.group(1)) if match else None

    def extract_str(pattern):
        match = re.search(rf"{model_id} has_(?:attribute|parsed_attribute) {pattern}:([^\n]+)", context)
        return match.group(1).strip() if match else None

    memory_speed = extract_value("max_memory_speed_mt_s")
    memory_channels = extract_value("memory_channels")
    memory_type = extract_str("memory_type")

    width = 8  # Default for DDR
    if memory_type and "DDR" in memory_type:
        width = 8

    return {
        "memory_speed_mt_s": memory_speed,
        "memory_channels": memory_channels,
        "memory_type": memory_type,
        "width_bytes": width
    }



def calculate_memory_bandwidth(attributes: Dict) -> Tuple[Dict, str]:
    """
    Calculate total memory bandwidth and return structured values for response generation.
    """
    speed = attributes.get("memory_speed_mt_s")
    channels = attributes.get("memory_channels")
    width = attributes.get("width_bytes", 8)

    if not speed or not channels:
        return None, "Missing memory_speed_mt_s or memory_channels. Cannot compute memory bandwidth."

    total_bandwidth = speed * width * channels / 1000  # GB/s

    return {
        "speed": speed,
        "width": width,
        "channels": channels,
        "total_bandwidth_gb_s": total_bandwidth
    }, None


def generate_memory_bandwidth_response(model_id: str, result: Dict) -> str:
    speed = result["speed"]
    width = result["width"]
    channels = result["channels"]
    total_bandwidth = result["total_bandwidth_gb_s"]

    return f"""
To calculate the memory bandwidth of {model_id}, we use the following formula:

Memory Bandwidth (GB/s) = memory_speed (MT/s) × memory_bus_width (Bytes) × number_of_channels ÷ 1000

Given:
- Memory Speed: {speed} MT/s
- Memory Bus Width: {width} Bytes
- Number of Channels: {channels}

Substituted:
Memory Bandwidth = {speed} × {width} × {channels} ÷ 1000 = {total_bandwidth:.1f} GB/s

Final Result:
The {model_id} processor provides a memory bandwidth of {total_bandwidth:.1f} GB/s.
""".strip()
