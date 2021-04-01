import torch
import os

from upstream.aalbert.system import PretrainedSystem as _PretrainedSystem

def audio_albert_local(ckpt, model_config, training_config, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _PretrainedSystem.load_from_checkpoint(checkpoint=ckpt, model_config=kwargs['model_config'], training_config=kwargs['training_config'], args={})
