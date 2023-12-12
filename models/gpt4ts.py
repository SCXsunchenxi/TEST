import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
from models.embedding import DataEmbedding
from encoders import wrapper
import json
import os

class gpt4ts_classification(nn.Module):

    def __init__(self, config, data):
        super(gpt4ts_classification, self).__init__()
        self.pred_len = 0
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        self.gpt_layers = 6
        self.feat_dim = data.feature_df.shape[1]
        self.num_classes = len(data.class_names)
        self.d_model = config['d_model']

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        # self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, config['d_model'], config['dropout'])

        self.gpt2 = GPT2Model.from_pretrained('./models/gpt2', output_attentions=True, output_hidden_states=True)
        # just use 6 layers (12->6)
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]

        # for i, (name, param) in enumerate(self.gpt2.named_parameters()):
        #     # tuning layernorm and positional embedding
        #     if 'ln' in name or 'wpe' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        # Freeze the gpt network
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            # tuning layernorm and positional embedding
            if 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # if config['gpu']=='-1':
        #     self.device = torch.device('cpu')
        # else:
        #     self.device = torch.device('cuda:{}'.format(config['gpu']))
        # self.gpt2.to(device=self.device)

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.ln_proj = nn.LayerNorm(config['d_model'] * self.patch_num)
        self.out_layer = nn.Linear(config['d_model'] * self.patch_num, self.num_classes)

        self.encoder = wrapper.CausalCNNEncoderClassifier()
        hf = open('./encoders/default_hyperparameters.json', 'r')
        hp_dict = json.load(hf)
        hf.close()
        hp_dict['cuda'] = False
        hp_dict['gpu'] = -1
        hp_dict['local_rank'] = -1
        hp_dict['out_channels'] = self.d_model
        self.encoder.set_params(**hp_dict)
        self.encoder.load(os.path.join(config['encoder_save_path'], config['data_dir'].split('/')[-1]))
        self.encoder.encoder.eval()

    def forward(self, x_enc,x_mark_enc):
        B, L, M = x_enc.shape # 64,1751,3

        input_x = rearrange(x_enc, 'b l m -> b m l')# 64,3,1751
        input_x = self.padding_patch_layer(input_x) # 64,3,1751
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride) # 64,3,219,8

        # Use exteral encorder -------
        input_x = rearrange(input_x, 'b m n p-> (b n) m p')# 64,3,219,8 -> 64x219,3,8
        outputs=self.encoder.encode(input_x.numpy()) # 64x219,3,8 -> 64x219,768
        outputs=torch.from_numpy(outputs).type(torch.FloatTensor)
        outputs = rearrange(outputs, '(b n) o-> b n o',b=B)# 64x219,768 -> 64,219,768

        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state # 64,219,768

        outputs = self.act(outputs).reshape(B, -1) # 64,219x768
        outputs = self.ln_proj(outputs) # 64,219x768
        outputs = self.out_layer(outputs) # 64,4

        return outputs


    def forward_wo_encoder(self, x_enc,x_mark_enc):
        B, L, M = x_enc.shape # 64,1751,3

        input_x = rearrange(x_enc, 'b l m -> b m l')# 64,3,1751
        input_x = self.padding_patch_layer(input_x) # 64,3,1751
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride) # 64,3,219,8
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')# 64,219,24

        outputs = self.enc_embedding(input_x, None)# 64,219,768 None is no positional embedding

        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state # 64,219,768

        outputs = self.act(outputs).reshape(B, -1) # 64,219x768
        outputs = self.ln_proj(outputs) # 64,219x768
        outputs = self.out_layer(outputs) # 64,4

        return outputs






class gpt4ts_forecasting(nn.Module):

    def __init__(self, config):
        super(gpt4ts_forecasting, self).__init__()
        self.is_gpt = config['is_gpt']
        self.patch_size = config['patch_size']
        self.pretrain = config['pretrain']
        self.stride = config['stride']
        self.patch_num = (config['seq_len'] - self.patch_size) // self.stride + 1
        self.d_model=config['d_model']
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.pred_len=config['pred_len']

        if self.is_gpt:
            if self.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('./models/gpt2', output_attentions=True,
                                                      output_hidden_states=True)
            else:
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:config['gpt_layers']]
            print("gpt2 = {}".format(self.gpt2))

        self.in_layer = nn.Linear(self.patch_size, self.d_model)
        self.out_layer = nn.Linear(self.d_model * self.patch_num, self.pred_len)

        # if config['freeze'] and self.pretrain:
        #     for i, (name, param) in enumerate(self.gpt2.named_parameters()):
        #         if 'ln' in name or 'wpe' in name:
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False

        # Freeze the gpt network
        if config['freeze'] and self.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.train()

        self.cnt = 0

        self.encoder = wrapper.CausalCNNEncoderClassifier()
        hf = open('./encoders/default_hyperparameters.json', 'r')
        hp_dict = json.load(hf)
        hf.close()
        hp_dict['cuda'] = False
        hp_dict['gpu'] = -1
        hp_dict['local_rank'] = -1
        hp_dict['out_channels'] = self.d_model
        self.encoder.set_params(**hp_dict)
        # self.encoder.load(os.path.join(config['encoder_save_path'], config['data_dir'].split('/')[-1]))
        self.encoder.encoder.eval()

    def forward_wo_encoder(self, x, it=-1):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l') # 512,512,1 -> 512,1,512

        x = self.padding_patch_layer(x)# 512,1,520
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)# 512,1,64,16
        x = rearrange(x, 'b m n p -> (b m) n p') # 512,64,16

        outputs = self.in_layer(x)# 512,64,768
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state  # 512,64,16

        outputs = self.out_layer(outputs.reshape(B * M, -1)) # 512,96
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B) # 512,96,1

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs

    def forward(self, x, it=-1):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l') # 512,512,1 -> 512,1,512

        x = self.padding_patch_layer(x)# 512,1,520
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)# 512,1,64,16

        # Use exteral encorder -------
        x = rearrange(x, 'b m n p-> (b n) m p') # 512x64,1,16
        outputs=self.encoder.encode(x.numpy()) # 512x64,1,16 -> 512x64,768
        outputs=torch.from_numpy(outputs).type(torch.FloatTensor)
        outputs = rearrange(outputs, '(b n) o-> b n o',b=B)# 512x64,768 -> 512,64,768

        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state  # 512,64,16

        outputs = self.out_layer(outputs.reshape(B * M, -1)) # 512,96
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B) # 512,96,1

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs


