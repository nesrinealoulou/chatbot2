# FP64 handlers
from calculations.fp64 import (
    calculate_fp64_performance,
    generate_fp64_response
)

# FP32 handlers
from calculations.fp32 import (
    calculate_fp32_performance,
    generate_fp32_response
)

# Shared per-core logic for both FP64 and FP32
from calculations.per_core import (
    calculate_per_core_performance,
    generate_per_core_response
)


# handlers/performance.py
def handle_fp64(model_id, context, attributes):
    results, err = calculate_fp64_performance(attributes)
    if err:
        return f"⚠️ FP64 error for {model_id}: {err}"
    response = generate_fp64_response(model_id, results)
    return response if response else f"⚠️ No FP64 response generated for {model_id}"


def handle_fp64_per_core(model_id, context, attributes):
    result, err = calculate_per_core_performance(attributes, precision="fp64")
    if err:
        return f"⚠️ FP64 per-core error for {model_id}: {err}"
    response = generate_per_core_response(model_id, result, precision="fp64")
    return response if response else f"⚠️ No FP64 per-core response generated for {model_id}"


# handlers/precision.py
def handle_fp32(model_id, context, attributes):
    results, err = calculate_fp32_performance(attributes)
    if err:
        return f"⚠️ FP32 error for {model_id}: {err}"
    response = generate_fp32_response(model_id, results)
    return response if response else f"⚠️ No FP32 response generated for {model_id}"


def handle_fp32_per_core(model_id, context, attributes):
    result, err = calculate_per_core_performance(attributes, precision="fp32")
    if err:
        return f"⚠️ FP32 per-core error for {model_id}: {err}"
    response = generate_per_core_response(model_id, result, precision="fp32")
    return response if response else f"⚠️ No FP32 per-core response generated for {model_id}"



