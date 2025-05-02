import torch

CONV_FEATURE_LAYERS = "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]"


def get_feat_extract_output_lengths(conv_feature_layers, input_lengths):
    """
    Computes the output length of the convolutional layers
    """

    def _conv_out_length(input_length, kernel_size, stride):
        return torch.floor((input_length - kernel_size) / stride + 1)

    conv_cfg_list = eval(conv_feature_layers)

    for i in range(len(conv_cfg_list)):
        input_lengths = _conv_out_length(
            input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
        )

    return input_lengths.to(torch.long)
