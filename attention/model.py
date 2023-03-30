from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

MAX_LENGTH = 10

# seq2seq 모델
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)

    def forward(self, input, hidden):
        # input = sequence length
        print(f'input={input.size()}')
        input = self.embedding(input) # [1, 256]
        embedded = input.view(1, 1, -1) # [1, 1, 256]
        output = embedded
        # GRU input must 3 dim. (sequence length, batch_size, input_size)
        # output=(seq_len, batch, num_directions * hidden_size)
        # hidden=(num_layers * num_directions, batch, hidden_size)
        output, hidden = self.gru(output, hidden) # [1, 1, 256], [1, 1, 256]

        print('=============== encoder ===============')
        print(f'input={input.size()}')
        print(f'embedded={embedded.size()}, output={output.size()}')
        print(f'hidden={hidden.size()}')
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
class AttnDecoderRNN(nn.Module):
    def __init__(
        self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # input=(batch_size, seq_length)
        # output=(batch_size, seq_length, hidden_size)
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        
        # input=(batch_size, hidden_size*2)
        # output=(batch_size, max_length)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        
        # input=(batch_size, hidden_size*2)
        # output=(batch_size, hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        
        # input/output=(seq_length, batch_size, hidden_size)
        self.gru = nn.GRU(
            input_size=self.hidden_size, hidden_size=self.hidden_size)
        
        # input=(batch_size, hidden_size)
        # output=(batch_size, output_size)        
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # input=[1, 1], hidden=[1, 1, 256], encoder_outputs=[10, 256]
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded) # [1, 1, 256]

        # [M, N, K], [M, N, K] cat dim=1 -> [M, N+N, K] 차원 더함
        # attn 레이어에 입력으로 사용하기 위해 마지막 차원 256 + 256
        concat_tensor = torch.cat(
            tensors=(embedded[0], hidden[0]), dim=1) # [1, 512]

        # max length = 10
        attn_weights = F.softmax(self.attn(concat_tensor), dim=1) # [1, 10]

        # Batch Matrix Multiplication = [B, N, M] x [B, M, P] = [B, N, P]
        # (1, 10) X (10, 256) = (1, 1, 10) X (1, 10, 256) = (1, 1, 256)
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0),
            encoder_outputs.unsqueeze(0)) # [1, 1 ,256]

        output = torch.cat((embedded[0], attn_applied[0]), 1) # [1, 512]
        output_unsq = self.attn_combine(output).unsqueeze(0) # [1, 1, 256]

        output_relu = F.relu(output_unsq) # [1, 1, 256]
        output, hidden = self.gru(
            output_relu, hidden) # [1, 1, 256], [1, 1, 256]

        output_sm = F.log_softmax(self.out(output[0]), dim=1) # [1, 47]
        
        # print('=============== decoder ===============')
        # print(f'embedded={embedded.size()}, concat_tensor={concat_tensor.size()}')
        # print(f'attn_weights={attn_weights.size()}, attn_applied={attn_applied.size()}')
        # print(f'output={output.size()}, output_unsq={output_unsq.size()}')
        # print(f'output_relu={output_relu.size()}, output={output.size()}')
        # print(f'hidden={hidden.size()}, output_sm={output_sm.size()}')
        return output_sm, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
