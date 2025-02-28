TINY_VIT_5M_ACTIVATIONS_OF_INTEREST = {
    "input": ["patch_embed"],
    "output": ["patch_embed"]
    + ["layers." + str(i) for i in range(4)]
    + ["layers." + str(i) + ".blocks." + str(j) for i in [0, 1, 3] for j in range(2)]
    + ["layers.2.blocks." + str(i) for i in range(6)],
}
