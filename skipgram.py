import numpy as np
import torch
import torch.nn.functional as F

corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'beijing is china capital',
    'berlin is germany capital',
    'paris is france capital',
]

def tokenizer_corpus(corpus_in):
  dict_word = []
  for line in corpus_in:
    dict_word += line.split()
  return sorted(list(set(dict_word)))


def wandids(dictword):
  w2id = {}
  id2w = {}
  for idx, w in enumerate(dictword):
    w2id[w] = idx
    id2w[idx] = w
  return w2id, id2w

def get_input_layer(word_idx):
  x = torch.zeros(vocab_lenth)
  x[word_idx] = 1
  return x

windows = 2
pairs = []
dict_word = tokenizer_corpus(corpus)
w2id, id2w = wandids(dict_word)
vocab_lenth = len(w2id)

for sentence in corpus:
  id_sentence = [w2id[w] for w in sentence.split()]
  for centw_pos in range(len(id_sentence)):
    for bias in range(-windows, windows + 1):
      context_w_pos = centw_pos + bias
      if context_w_pos < 0 or context_w_pos >= len(id_sentence) or context_w_pos == centw_pos:
        continue
      pairs.append((id_sentence[context_w_pos], id_sentence[centw_pos]))
pairs = np.array(pairs)


embeding_size = 5

W1 = torch.rand((vocab_lenth, embeding_size), requires_grad=True)
W2 = torch.rand((embeding_size, vocab_lenth), requires_grad=True)


epoch = 2000
learning_rate = 0.01

for idx in range(epoch):
  loss_val = 0
  for data, label in pairs:
    x = get_input_layer(data)
    y_true = torch.from_numpy(np.array([label]))
    z1 = torch.matmul(x, W1)
    z2 = torch.matmul(z1, W2)
    log_softmax = F.log_softmax(z2, dim=0)
    loss = F.nll_loss(log_softmax.view(1,-1), y_true)
    loss_val += loss.data
    loss.backward()
    W1.data -= learning_rate * W1.grad.data
    W2.data -= learning_rate * W2.grad.data
    W1.grad.data.zero_()
    W2.grad.data.zero_()
  if idx % 10 == 0:    
      print(f'Loss at epo {idx}: {loss_val/len(pairs)}')

context_to_predict = get_word_vector_v('paris')
hidden = torch.from_numpy(context_to_predict)
a = torch.matmul(hidden, W2)
probs = F.softmax(a, dim=0).data.numpy()
for context, prob in zip(dict_word, probs):
    print(f'{context}: {prob:.10f}')