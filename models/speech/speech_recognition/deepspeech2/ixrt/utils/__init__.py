from .load_tensorrt import deepspeech2_trtapi_ixrt, setup_io_bindings


VOCABLIST = ['<blank>',
             '<unk>',
             "'",
             '<space>',
             'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
             '<eos>']


def check_target(inference, target):
    satisfied = False
    if inference > target:
        satisfied = True
    return satisfied
