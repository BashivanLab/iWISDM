import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import math

class TFEncoder(nn.Module):
    def __init__(self, hidden_size, img_encoder, device, dim_transformer_ffl=2048, nhead = 16, blocks=2, output_size = 3, max_frames=6):
        super().__init__()

        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()
            return hook

        self.device = device

        self.max_frames = max_frames

        # set up the CNN model
        self.cnnmodel = img_encoder
        # freeze layers of cnn model
        for paras in self.cnnmodel.parameters():
            paras.requires_grad = False
        
        self.cnnmodel.layer4[2].relu.register_forward_hook(get_activation('relu'))
        self.cnnlayer = torch.nn.Conv2d(2048, hidden_size, 1)

        self.input_size = hidden_size*7*7
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.in2hidden = nn.Linear(self.input_size, hidden_size)
        self.layer_norm_in = nn.LayerNorm(self.hidden_size)
        
        self.pos_emb = PositionalEncoding(hidden_size, self.max_frames)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, dim_feedforward=dim_transformer_ffl, nhead=nhead, batch_first=False)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=blocks)

        self.hidden2output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, frames, instructions=None):

        self.batch_size = frames.shape[0]
        self.seq_len = frames.shape[1]

        causal_mask = self.generate_causal_mask(self.seq_len)
        padding_mask = None # self.generate_pad_mask(frames)
        
        x = torch.swapaxes(frames, 0, 1).float() # (seq_len, batchsize, nc, w, h)

        x_acts = []
        for i in range(self.seq_len):
            temp = self.cnnmodel(x[i,:,:,:,:])
            x_act = self.cnnlayer(self.activation["relu"])
            x_acts.append(x_act) # (batchsize, nc, w, h) = (batchsize, 2048, 7,7)

        x_acts = torch.stack(x_acts, axis = 0) # (seqlen, batchsize,nc, w,h) 
        x_acts = x_acts.reshape(x_acts.shape[0], self.batch_size, -1) # flatten nc,w,h into one dim
         
        hidden_x = self.layer_norm_in(self.pos_emb(self.in2hidden(x_acts.float())))
        encoder_output = self.encoder(hidden_x, mask=causal_mask, src_key_padding_mask=padding_mask)
        out = self.hidden2output(encoder_output)
        
        return out
        
    def init_hidden(self, batch_size):
        return nn.init.kaiming_uniform_(torch.empty(1, batch_size, self.hidden_size))

    # Creates a square Sequential/Causal mask of size sz
    def generate_causal_mask(self, sz: int) -> Tensor:
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(self.device)

    # Generates a padding masks for each sequence in a batch
    def generate_pad_mask(self, batch):

        pad_tensor = torch.ones((batch.shape[2])).to(self.device)

        mask = np.zeros((batch.shape[0],batch.shape[1]))

        for s in range(0, batch.shape[0]):
            for v in range(0, batch[s].shape[0]):
                new_s = torch.all(batch[s][v] == pad_tensor)
                mask[s][v] = new_s

        return torch.tensor(mask).bool().to(self.device)

# Sinusoidal Positional Encoding Module
class PositionalEncoding(nn.Module):
    # Positional encoding module taken from PyTorch Tutorial
    # Link: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x



class RNN(nn.Module):
    def __init__(self, hidden_size, img_encoder, device, output_size = 3, max_frames=6):
        super().__init__()
        
        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()
            return hook

        self.device = device

        self.max_frames = max_frames

        # set up the CNN model
        self.cnnmodel = img_encoder
        # freeze layers of cnn model
        for paras in self.cnnmodel.parameters():
            paras.requires_grad = False
        
        self.cnnmodel.layer4[2].relu.register_forward_hook(get_activation('relu'))
        self.cnnlayer = torch.nn.Conv2d(2048, hidden_size, 1)

        self.input_size = hidden_size*7*7
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.in2hidden = nn.Linear(self.input_size, hidden_size)
        self.layer_norm_in = nn.LayerNorm(self.hidden_size)
        
        self.rnn = nn.RNN(
            input_size = self.hidden_size, 
            hidden_size = self.hidden_size,
            # nonlinearity = "relu",  #  DIFFERENCE
            )

        self.layer_norm_rnn = nn.LayerNorm(self.hidden_size)
        self.hidden2output = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input_img, instructions=None, hidden_state = None):

        self.batch_size = input_img.shape[0]
        self.seq_len = input_img.shape[1]
        
        x = torch.swapaxes(input_img, 0, 1).float() # (seq_len, batchsize, nc, w, h)
        
        x_acts = []
        for i in range(self.seq_len):
            temp = self.cnnmodel(x[i,:,:,:,:])
            x_act = self.cnnlayer(self.activation["relu"])
            x_acts.append(x_act) # (batchsize, nc, w, h) = (batchsize, 2048, 7,7)
            
        x_acts = torch.stack(x_acts, axis = 0) # (seqlen, batchsize,nc, w,h) 
        
        x_acts = x_acts.reshape(x_acts.shape[0], self.batch_size, -1) # flatten nc,w,h into one dim
         
        if hidden_state == None:
            self.hidden_state = self.init_hidden(batch_size = self.batch_size)
        hidden_x = self.layer_norm_in(torch.relu(self.in2hidden(x_acts.float()))).to(self.device)
        rnn_output, _ = self.rnn(hidden_x, self.hidden_state.to(self.device))
        rnn_output = self.layer_norm_rnn(rnn_output)
        out = self.hidden2output(torch.tanh(rnn_output))
        
        return out
        
    def init_hidden(self, batch_size):
        return nn.init.kaiming_uniform_(torch.empty(1, batch_size, self.hidden_size))



class InsEncoder(object):
    def __init__(self, model, tokenizer, device):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, instruction):
        instruction = self.tokenizer(instruction, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        #Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Compute token embeddings
        with torch.no_grad():
            lm_output = self.model(**instruction)

        # Perform pooling
        sentence_embeddings = mean_pooling(lm_output, instruction['attention_mask'])
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            
        return sentence_embeddings

