import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from model.mae_vit import mae_vit_base_patch16
from model.mae_bert import BertConfig, BertForMaskedLM

from transformers import BertTokenizer
from sklearn.metrics import accuracy_score,roc_auc_score

label_map = {
    'AD':2,
    'Dementia':2,
    'Demented':2,
    'MCI':1,
    'Impaired-not-MCI':1,
    'CN':0,
    'Control':0,
    'Nondemented':0,
}


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class AlifuseModel(nn.Module):
    def __init__(
        self,
        max_txt_len=70,
        hidden_size=768,
    ):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side='right')
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        self.tokenizer.enc_token_id = self.tokenizer.additional_special_tokens_ids[0]

        self.vision_transformer = mae_vit_base_patch16()

        config = BertConfig.from_pretrained("bert-base-uncased")
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = 24
        # config
        self.text_transformer = BertForMaskedLM.from_pretrained("bert-base-uncased", config=config)
        self.text_transformer.resize_token_embeddings(len(self.tokenizer))

        self.vision_proj = nn.Linear(self.vision_transformer.dim, hidden_size)
        self.text_proj = nn.Linear(self.text_transformer.config.hidden_size, hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(768*2, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = max_txt_len
        self.mlm_probability = 0.5
        self.criterion = nn.CrossEntropyLoss()


    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.bos_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        #  we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 1.)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
        
    def forward(self, samples):
        # 

        image,text,label = samples
        image = image.unsqueeze(1).cuda() # bs c 128 128 128
        label = label.cuda()
        image_embeds,mask,ids_restore = self.vision_transformer.forward_encoder(
            img = image,
            text_emb=None,
            mask_ratio=0)
        image_embeds = self.vision_proj(image_embeds)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        image_feat = F.normalize(image_embeds[:,0,:], dim=-1)

        text_tokens = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_txt_len, return_tensors="pt",).to(image.device) # inputs_ids: bs 70

        # 

        encoded_layers, _, self_attn, cross_attn = self.text_transformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            output_attentions=True)
        text_embeds = self.text_proj(encoded_layers[-1]) # bs txt_len 768
        text_feat = F.normalize(text_embeds[:, 0, :], dim=-1)

        ###============== Image-text Contrastive ===================###
        # image-text similarity
        sim_i2t = image_feat @ text_feat.T / self.temp
        sim_t2i = text_feat @ image_feat.T / self.temp
        bs = image.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(image.device)
        loss_itc = (F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                    + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)) / 2

        ##================= MLM ========================##

        # text restore
        input_ids = text_tokens.input_ids.clone()
        input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)  
        input_ids, labels = self.mask(input_ids, self.text_transformer.config.vocab_size, image.device, targets=labels,probability_matrix = probability_matrix)
        lm_output = self.text_transformer(
            input_ids,
            attention_mask=text_tokens.attention_mask,
            encoder_hidden_states=image_embeds,
            masked_lm_labels=labels,
            output_attentions=True,
        )
        text_res_loss,_ = lm_output


        # image restore
        image_res_loss,_,_ = self.vision_transformer(
            image,
            text_emb=text_embeds,
            mask_ratio=0.5)

        
        ##================= CLS ========================##
        h_concat = torch.cat([image_embeds[:,0], text_embeds[:,0]], dim=-1)
        y_hat = self.mlp(h_concat)
        loss_cls = self.criterion(y_hat, label)


        loss=loss_itc+(text_res_loss+image_res_loss)+loss_cls
        loss=(text_res_loss+image_res_loss)+loss_cls
        print(' loss_itc: ', loss_itc.item(), ' text_res_loss: ',text_res_loss.item(), ' image_res_loss: ',image_res_loss.item(), ' loss_cls: ',loss_cls.item())

        return {"loss": loss}
  
    def predict(self, samples):
        image,text,label = samples
        image = image.unsqueeze(1).cuda() # bs c 128 128 128
        label = label.cuda()
        image_embeds,mask,ids_restore = self.vision_transformer.forward_encoder(
            img = image,
            text_emb=None,
            mask_ratio=0)
        image_embeds = self.vision_proj(image_embeds)


        text_tokens = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_txt_len, return_tensors="pt",).to(image.device) # inputs_ids: bs 70
        encoded_layers, _, self_attn, cross_attn = self.text_transformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            output_attentions=True)
        text_embeds = self.text_proj(encoded_layers[-1]) # bs txt_len 768        
        ##================= CLS ========================##
        h_concat = torch.cat([image_embeds[:,0], text_embeds[:,0]], dim=-1)
        y_hat = self.mlp(h_concat)

        return y_hat
       