from copy import deepcopy

import torch
import torch.nn as nn


class EMA(nn.Module):
    def __init__(
        self, 
        model,
        ema_update_every=10,
        ema_decay=0.995,
        ema_update_after_step=100,
        inverse_gamma=1.0,
        power=2/3,
    ):
        super(EMA, self).__init__()
        self.model = model
        self.model.gru.flatten_parameters()
        self.ema_model = deepcopy(self.model)
        self.ema_model.gru.flatten_parameters()
        self.ema_model.requires_grad_(False)
        
        self.parameter_names = {name for name, param in self.model.named_parameters() if param.dtype in [torch.float, torch.float16]}
        
        self.ema_update_after_step = ema_update_after_step
        self.ema_update_every = ema_update_every
        
        self.ema_decay = ema_decay
        self.inverse_gamma = inverse_gamma
        self.power = power
        
        self.step = 0
        self.inited = False
        
    def update(self):
        step = self.step
        self.step += 1
        if (step % self.ema_update_every) != 0:
            return
        if step <= self.ema_update_after_step:
            self.copy_params()
            return
        if not self.inited:
            self.copy_params()
            self.inited = True
        self.update_moving_average(self.ema_model, self.model)
        
    def get_params(self, model):
        for name, param in model.named_parameters():
            if name in self.parameter_names:
                yield name, param
        
    def copy_params(self):
        for (_, ema_param), (_, param) in zip(self.get_params(self.ema_model), self.get_params(self.model)):
            ema_param.data.copy_(param.data)
            
    @torch.no_grad()
    def update_moving_average(self, ema_model, model):
        current_decay = self.get_current_decay()
        for (_, ema_param), (_, param) in zip(self.get_params(ema_model), self.get_params(model)):
            ema_param.data.lerp_(param.data, 1 - current_decay)
    
    def get_current_decay(self):
        epoch = max(0, self.step - self.ema_update_after_step - 1)
        value = 1 - (1 + epoch / self.inverse_gamma) ** (-self.power)
        if epoch <= 0:
            return 0.0
        else:
            value = max(0.0, value)
            value = min(self.ema_decay, value)
            return value