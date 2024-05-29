import numpy as np

num_parameters = {
    "Gemma 2B": 524_550_144 + 1_981_884_416,  # Source: Paper
    "Gemma 7B": 786_825_216 + 7_751_248_896,  # Source: Paper
    "GPT 3.5 Turbo": 20_000_000_000,  # Source: Undisclosed ... urban legend on the internet says 20B
    "GPT 4": 1_760_000_000_000,  # Source: Undisclosed ... estimate from https://the-decoder.com/gpt-4-architecture-datasets-costs-and-more-leaked/
    "GPT 4 Turbo Preview": np.nan,  # Source: Undisclosed
    "Llama 2 7B": 7_000_000_000,  # Source: Name
    "Llama 2 13B": 13_000_000_000,  # Source: Name
    "Llama 2 70B": 70_000_000_000,  # Source: Name
    "Mistral 7B v0.2": 7_000_000_000,  # Source: Name
    "Mixtral 8x7B": 46_700_000_000,  # Source: https://docs.mistral.ai/models/#:~:text=Mixtral%208X7B%20is%20a%20sparse,the%20cost%20of%20more%20vRAM.
    "Orca 2 7B": 7_000_000_000,  # Source: Name + Llama 2 7B
    "Orca 2 13B": 13_000_000_000,  # Source: Name + Llama 2 13B
    "Starling LM 7B Beta": 7_000_000_000,  # Source: Name
    "Vicuna 7B 16K": 7_000_000_000,  # Source: Name + Llama 2 7B
}

# ELO ratings on 08 April 2024 (last update 29 March 2024) from https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
elo_ratings = {
    "Gemma 2B": 996,
    "Gemma 7B": 1037,
    "GPT 3.5 Turbo": 1098,  # 0125 checkpoint
    "GPT 4": 1160,  # 0613 checkpoint
    "GPT 4 Turbo Preview": 1249,  # 0125 checkpoint
    "Llama 2 7B": 1031,
    "Llama 2 13B": 1046,
    "Llama 2 70B": 1083,
    "Mistral 7B v0.2": 1072,
    "Mixtral 8x7B": 1114,
    "Orca 2 7B": np.nan,
    "Orca 2 13B": np.nan,
    "Starling LM 7B Beta": 1118,
    "Vicuna 7B 16K": 1002,  # For the non-16K version
}

mt_bench_scores = {
    "Gemma 2B": np.nan,
    "Gemma 7B": np.nan,
    "GPT 3.5 Turbo": 7.94,  # (for 0314 version rather than 0125) ... Source: LMSYS Chatbot Arena Leaderboard
    "GPT 4": 9.18,  # Source: LMSYS Chatbot Arena Leaderboard
    "GPT 4 Turbo Preview": 9.32,  # (for 1106 version rather than 0125) ... Source: LMSYS Chatbot Arena Leaderboard
    "Llama 2 7B": 6.27,  # Source: LMSYS Chatbot Arena Leaderboard
    "Llama 2 13B": 6.65,  # Source: LMSYS Chatbot Arena Leaderboard
    "Llama 2 70B": 6.86,  # Source: LMSYS Chatbot Arena Leaderboard
    "Mistral 7B v0.2": 7.6,  # Source: LMSYS Chatbot Arena Leaderboard
    "Mixtral 8x7B": 8.3,  # Source: LMSYS Chatbot Arena Leaderboard
    "Orca 2 7B": 5.65,  # Source: Paper ... Table 3 Average
    "Orca 2 13B": 6.15,  # Source: Paper ... Table 3 Average
    "Starling LM 7B Beta": 8.12,  # Source: LMSYS Chatbot Arena Leaderboard
    "Vicuna 7B 16K": 6.22,  # Source: LMSYS Chatbot Arena Leaderboard
}

mmlu_scores = {
    "Gemma 2B": 42.3,  # Source: LMSYS Chatbot Arena Leaderboard
    "Gemma 7B": 64.3,  # Source: LMSYS Chatbot Arena Leaderboard
    "GPT 3.5 Turbo": 70.0,  # (for 0314 version rather than 0125) ... Source: LMSYS Chatbot Arena Leaderboard
    "GPT 4": 86.4,  # (for 0314 version rather than 0613) ... Source: LMSYS Chatbot Arena Leaderboard
    "GPT 4 Turbo Preview": np.nan,
    "Llama 2 7B": 45.8,  # Source: LMSYS Chatbot Arena Leaderboard
    "Llama 2 13B": 53.6,  # Source: LMSYS Chatbot Arena Leaderboard
    "Llama 2 70B": 63.0,  # Source: LMSYS Chatbot Arena Leaderboard
    "Mistral 7B v0.2": 55.4,  # (for v0.1 rather than v0.2) ... Source: LMSYS Chatbot Arena Leaderboard
    "Mixtral 8x7B": 70.6,  # Source: LMSYS Chatbot Arena Leaderboard
    "Orca 2 7B": 53.7,  # Source: Paper
    "Orca 2 13B": 57.73,  # Source: Paper
    "Starling LM 7B Beta": 63.9,  # (for alpha version rather than beta) ... Source: LMSYS Chatbot Arena Leaderboard
    "Vicuna 7B 16K": 48.5,  # Source: LMSYS Chatbot Arena Leaderboard
}

hellaswag_scores = {
    "Gemma 2B": 71.4,  # Source: Paper
    "Gemma 7B": 81.2,  # Source: Paper Table 6
    "GPT 3.5 Turbo": 85.5,  # Source: HellaSwag blog post ... points to GPT 4 Technical Report ... so probalby different checkpoint
    "GPT 4": 95.3,  # Source: HellaSwag blog post ... points to GPT 4 Technical Report ... so probalby different checkpoint
    "GPT 4 Turbo Preview": np.nan,
    "Llama 2 7B": 77.2,  # Source: Paper
    "Llama 2 13B": 80.7,  # Source: Paper
    "Llama 2 70B": 85.3,  # Source: HellaSwag blog post ... points to Llama 2 paper
    "Mistral 7B v0.2": 81.3,  # Source: HellaSwag blog post ... points to Mistral paper
    "Mixtral 8x7B": 84.4,  # Source: Paper
    "Orca 2 7B": 81.56,  # Source: Paper ... Figure 5
    "Orca 2 13B": 77.75,  # Source: Paper ... Figure 5
    "Starling LM 7B Beta": np.nan,
    "Vicuna 7B 16K": np.nan,
}
