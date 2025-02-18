VIT_B_ACTIVATIONS_OF_INTEREST = {
    "input": ["patch_embed"],
    "output": ["patch_embed"] + ["blocks." + str(i) for i in range(12)],
}
