import numpy as np

def get_masked_input_and_labels(texts, mask_token_id, mask_ratio=0.15):
    inp_mask = np.random.rand(*texts.shape) < mask_ratio
    # Set targets to -1 by default, it means ignore
    inp_mask[texts <= 2] = False
    # Set targets to -1 by default, it means ignore
    labels = -1 * np.ones(texts.shape, dtype=int)
    # Set labels for masked tokens
    labels[inp_mask] = texts[inp_mask]

    # Prepare input
    encoded_texts_masked = np.copy(texts)
    # Set input to [MASK] which is the last token for the 90% of tokens
    # This means leaving 10% unchanged
    inp_mask_2mask = inp_mask & (np.random.rand(*texts.shape) < 0.90)
    encoded_texts_masked[
        inp_mask_2mask
    ] = mask_token_id  # mask token is the last in the dict

    # Set 10% to a random token
    # inp_mask_2random = inp_mask_2mask & (np.random.rand(*texts.shape) < 1 / 9)
    # encoded_texts_masked[inp_mask_2random] = np.random.randint(
    #     3, mask_token_id, inp_mask_2random.sum()
    # )

    # Prepare sample_weights to pass to .fit() method
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0

    # y_labels would be same as texts i.e input tokens
    y_labels = np.copy(texts)

    return encoded_texts_masked, y_labels, sample_weights