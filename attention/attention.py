from __future__ import unicode_literals, print_function, division
from io import open
import time
import math
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from model import EncoderRNN, AttnDecoderRNN
from model import device as device
from loader import tensorsFromPair, tensorFromSentence, prepareData

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

MAX_LENGTH = 10

# 단어 -> 색인, 색인 -> 단어 사전
# 희귀 단어를 대체할 때 사용할 각 단어의 빈도를 가진 핼퍼 클래스
SOS_token = 0
EOS_token = 1

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

# 남은 예상 시간을 출력
def timeSince(since, percent):

    # 현재 시간과 진행률%을 고려해 경과된 시간
    def asMinutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

teacher_forcing_ratio = 0.5

def train(
    input_tensor, target_tensor,
    encoder, decoder, encoder_optimizer, decoder_optimizer,
    criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0] # output 값 갱신

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True
    else:
        use_teacher_forcing = False

    use_teacher_forcing = True
    if use_teacher_forcing:
        # Teacher forcing 포함: 목표를 다음 입력으로 전달
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Teacher forcing 미포함: 자신의 예측을 다음 입력으로 사용
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            # 입력으로 사용할 부분을 히스토리에서 분리
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(
    encoder, decoder, 
    n_iters, print_every=1000, 
    plot_every=100, learning_rate=0.01):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # print_every 마다 초기화
    plot_loss_total = 0  # plot_every 마다 초기화

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [
        tensorsFromPair(
            input_lang,
            output_lang,
            random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(
            input_tensor, target_tensor, encoder,
            decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (
                timeSince(start, iter / n_iters), iter, 
                iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # showPlot(plot_losses)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]
    
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

hidden_size = 256

# input_lang.n_words : 65
# input_lang.word2count : 63
# input_lang.word2index : 63
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)

# output_lang.n_words : 47
# output_lang.word2count : 45
# output_lang.word2index : 45
attn_decoder1 = AttnDecoderRNN(
    hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

# # encoder
# dummy_data = torch.rand(1).long().to(device)
# encoder_hidden = encoder1.initHidden()
# torch.onnx.export(
#     encoder1, (dummy_data, encoder_hidden), 'attn.onnx',
#     input_names=['input'], output_names=['output'])

# # decoder
# decoder_input = torch.rand(1, 1).long().to(device)
# decoder_hidden = attn_decoder1.initHidden()
# encoder_output = torch.rand(10, 256, dtype=torch.float32).to(device)

# torch.onnx.export(
#     attn_decoder1, (decoder_input, decoder_hidden, encoder_output),
#     'attn_decoder.onnx',
#     input_names=['input'], output_names=['output'])

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
# trainIters(encoder1, attn_decoder1, 1, print_every=5000)

# evaluateRandomly(encoder1, attn_decoder1)
