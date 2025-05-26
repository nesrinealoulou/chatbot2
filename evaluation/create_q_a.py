import json

questions = [
    "Does Xeon 8468H support PCIe Gen 5.0? Explain based on the context only.",
    "Which processor has faster RAM: 6530 or 6530P?",
    "Do the 7453 and 7573X have the same RAM speed support?",
    "List the PCIe lanes, AVX support, and SIMD units for the EPYC 947F",
    "Which CPU has better memory bandwidth and FP64 performance: EPYC 9755 or Xeon 9470?",
    "Does the EPYC 947F support ECC and how many memory channels does it have?",
    "What is the base clock, boost clock, and number of PCIe lanes of the EPYC 96846?",
    "Does the EPYC 9755 support AVX-512 and what is its memory type?",
    "Which chip supports newer interconnect standards: 6726P-B or EPYC 6546P-B? ",
    "comapare Xeonn 9564 vs Epycc 9754?"
    
]

mock_responses = [
    "Yes, the Intel Xeon Platinum 8468H supports PCIe Gen 5.0",
    "Both the EPYC 6530 support DDR5 memory, with a memory speed of 4800 MT/s , however EPYC 6530P do not exist",
    "Yes, both EPYC 7453 and EPYC 7573X support DDR4-3200 memory.",
    "EPYC 9474F has, 128 lanes of PCIe Gen 5 , Supports AVX2 and AVX-512, and 8 simds units",
    "EPYC 9755 has better memory bandwidth (12 channels vs 8) , i don't have enough context for fp64 performance",
    "Yes, the EPYC 9474F supports ECC memory ",
    "EPYC 9684X Base clock is 2.55 GHz and Boost clock: ~3.7 GHz and PCIe lanes: 128 (PCIe Gen 5.0)",
    "yes EPYC 9755 supports  AVX-512 and DDR5 memory",
    "Xeon 6726P-B support Gen 4 however EPYC 6546P-B does not exist",
    "intel xeon 9564 does not exist , however The AMD EPYC 9754 is a server/workstation processor with 128 cores, launched in June 2023, at an MSRP of $11900. It is part of the EPYC lineup, using the Zen 4c (Bergamo) architecture with Socket SP5."
    
]

# Combine into list of dicts
qa_db = [{"question": q, "expected_answer": a} for q, a in zip(questions, mock_responses)]

# Write to JSON file
with open("q_a_db.json", "w", encoding="utf-8") as f:
    json.dump(qa_db, f, indent=2, ensure_ascii=False)

print("✅ q_a_db.json has been created.")
