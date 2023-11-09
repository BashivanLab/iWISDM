
class DecoderTF(nn.Module):
    def __init__(self, ins_size, hidden_size, dim_transformer_ffl=2048, nhead = 16, blocks=2, output_size = 3,):
        super().__init__()

        # set up the CNN model
        self.cnnmodel = torch.load(IMGM_PATH, map_location=device)
        # freeze layers of cnn model
        for paras in self.cnnmodel.parameters():
            paras.requires_grad = False
        
        self.cnnmodel.layer4[2].relu.register_forward_hook(get_activation('relu'))
        self.cnnlayer = torch.nn.Conv2d(2048, hidden_size, 1)


        self.ins_input_size = ins_size
        self.img_input_size = hidden_size*7*7
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.insin2hidden = nn.Linear(self.ins_input_size, hidden_size)
        self.imgin2hidden = nn.Linear(self.img_input_size, hidden_size)
        self.layer_norm_in = nn.LayerNorm(self.hidden_size)
        
        self.pos_emb = PositionalEncoding(hidden_size, MAX_FRAMES)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, dim_feedforward=dim_transformer_ffl, nhead=nhead, batch_first=False)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=blocks)

        self.hidden2output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_img, input_ins, hidden_state = None, is_noise = False, causal_mask=None, padding_mask=None):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook


        self.batch_size = input_img.shape[0]
        self.seq_len = input_img.shape[1]
        
        img_x = torch.swapaxes(input_img, 0, 1).float() # (seq_len, batchsize, nc, w, h)
        
        img_x_acts = []
        for i in range(self.seq_len):
            temp = self.cnnmodel(img_x[i,:,:,:,:])
            x_act = self.cnnlayer(activation["relu"])
            img_x_acts.append(x_act) # (batchsize, nc, w, h) = (batchsize, 2048, 7,7)

        img_x_acts = torch.stack(img_x_acts, axis = 0) # (seqlen, batchsize,nc, w,h) 
        img_x_acts = img_x_acts.reshape(img_x_acts.shape[0], self.batch_size, -1) # flatten nc,w,h into one dim
         
        hidden_ins_x = self.layer_norm_in(self.insin2hidden(ins_x_acts.float()))
        hidden_img_x = self.layer_norm_in(self.pos_emb(self.imgin2hidden(img_x_acts.float())))
        decoder_output = self.decoder(hidden_img_x, hidden_ins_x, tgt_mask=mask, tgt_key_padding_mask=padding_mask)
        out = self.hidden2output(decoder_output)
        
        return out
        
    def init_hidden(self, batch_size):
        return nn.init.kaiming_uniform_(torch.empty(1, batch_size, self.hidden_size))

# Sinusoidal Positional Encoding Module
class PositionalEncoding(nn.Module):
    # Positional encoding module taken from PyTorch Tutorial
    # Link: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, max_len: int = MAX_FRAMES):
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

# Creates a square Sequential/Causal mask of size sz
def generate_causal_mask(sz: int) -> Tensor:
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# Generates a padding masks for each sequence in a batch
def generate_pad_mask(batch):

    pad_tensor = torch.ones((batch.shape[2])).to(device)

    mask = np.zeros((batch.shape[0],batch.shape[1]))

    for s in range(0, batch.shape[0]):
        for v in range(0, batch[s].shape[0]):
            new_s = torch.all(batch[s][v] == pad_tensor)
            mask[s][v] = new_s

    return torch.tensor(mask).bool().to(device)