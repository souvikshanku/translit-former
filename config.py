
class LM_Config:
    context_length = 64
    vocab_size = 140

    d_model = 32
    d_hidden = 64
    num_heads = 4
    d_head = 8  # d_model // num_heads
    num_block = 3

    use_prefix_lm_masking = True
