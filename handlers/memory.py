# Memory bandwidth handler
from calculations.memory_bw import (
    extract_memory_attributes,
    calculate_memory_bandwidth,
    generate_memory_bandwidth_response
)

# handlers/memory.py
def handle_memory_bandwidth(model_id, context, attributes):
    attributes = extract_memory_attributes(context, model_id)
    results, err = calculate_memory_bandwidth(attributes)
    if err:
        return f"⚠️ Memory bandwidth error for {model_id}: {err}"
    response = generate_memory_bandwidth_response(model_id, results)
    return response if response else f"⚠️ No memory bandwidth response generated for {model_id}"
