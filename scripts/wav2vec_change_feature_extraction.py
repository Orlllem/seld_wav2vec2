

# !wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt
# !wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt


import gc
from collections import namedtuple

import fairseq
import torch
from omegaconf.dictconfig import DictConfig

user_dir = "/home/seld_wav2vec2/src/seld_wav2vec2"
Arg = namedtuple("Arg", ["user_dir"])
arg = Arg(user_dir.__str__())
fairseq.utils.import_user_module(arg)

cp = "wav2vec_small.pt"
# cp = "wav2vec_vox_new.pt"


cp_save = "w2v_audio_base_4ch_unorm.pt"  # wav2vec_small.pt
# cp_save = "w2v_audio_large_4ch_norm.pt" # wav2vec_vox_new.pt

model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([
                                                                         cp], strict=True)
model = model[0]
model.eval()


w2v_state = model.state_dict()
content = dict(cfg.model)

res = DictConfig(
    content={**content, **{'in_channels': 4, 'in_conv_groups': 1}})

res._name = 'wav2vec2_ch'


model_4ch = task.build_model(res, from_checkpoint=True)


del model

state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(cp)


state_dict = model_4ch.state_dict()
state_model = state["model"].copy()
for key in state["model"]:
    if key in state_dict.keys():
        if state["model"][key].shape != state_dict[key].shape:
            print("key {} is not matching".format(key))
            state_model.pop(key)

model_4ch.load_state_dict(state_model, strict=False)


state['args'].arch = 'wav2vec2_ch'
state['model'] = model_4ch.state_dict()
state['cfg']['model'] = res


torch.save(state, cp_save)


# Load

state_new = fairseq.checkpoint_utils.load_checkpoint_to_cpu(cp_save)


del state_new

gc.collect()


model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([
                                                                         cp_save], strict=True)
model = model[0]
model.eval()


saved_4ch_w2v_state = model.state_dict()


for key in w2v_state.keys():
    if not torch.equal(w2v_state[key], saved_4ch_w2v_state[key]):
        print("key", key)


assert not torch.equal(w2v_state['feature_extractor.conv_layers.0.0.weight'],
                       saved_4ch_w2v_state['feature_extractor.conv_layers.0.0.weight'])
assert torch.equal(w2v_state['feature_extractor.conv_layers.1.0.weight'],
                   saved_4ch_w2v_state['feature_extractor.conv_layers.1.0.weight'])
assert torch.equal(w2v_state['feature_extractor.conv_layers.2.0.weight'],
                   saved_4ch_w2v_state['feature_extractor.conv_layers.2.0.weight'])
assert torch.equal(w2v_state['feature_extractor.conv_layers.3.0.weight'],
                   saved_4ch_w2v_state['feature_extractor.conv_layers.3.0.weight'])
