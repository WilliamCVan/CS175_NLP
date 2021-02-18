import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator, TabularDataset
import spacy
import random
from torchtext.data.metrics import bleu_score
from encDecoderLSTM import EncoderRNN, DecoderRNN

# python -m spacy download ja_core_news_sm
# python -m spacy download en_core_web_sm

spacy_japanese = spacy.load("ja_core_news_sm")
spacy_english = spacy.load("en_core_web_sm")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_japanese(text):
  #return [token.text for token in spacy_japanese.tokenizer(text)]
  return text.split(" ")

def tokenize_english(text):
  #return [token.text for token in spacy_english.tokenizer(text)]
  return text.split(" ")


# sample_text = "I love machine learning"
# print(tokenize_english(sample_text))
#
# jap_text = "これは文章です。"
# # doc = spacy_japanese("これは文章です。")
# # print([(w.text, w.pos_) for w in doc])
# print(tokenize_japanese(jap_text))


JAPANESE = Field(tokenize=tokenize_japanese,
               lower=True,
               init_token="<sos>",
               eos_token="<eos>")

ENGLISH = Field(tokenize=tokenize_english,
               lower=True,
               init_token="<sos>",
               eos_token="<eos>")


# https://dzlab.github.io/dltips/en/pytorch/torchtext-datasets/

fields = [
  ('eng', ENGLISH),
  ('jap', JAPANESE)
]

# load the dataset in json format
train_ds, valid_ds = TabularDataset.splits(
   path = 'datafiles',
   train = 'train_de-eng.tsv',
   validation = 'test.tsv',
   format = 'tsv',
   fields = fields,
   skip_header = False
)

# check an example
print(vars(train_ds[0]))
print()

print(f"Number of training examples: {len(train_ds.examples)}")
print(f"Number of validation examples: {len(valid_ds.examples)}")
print(train_ds[5].__dict__.keys())
print(train_ds[5].__dict__.values())
print()


JAPANESE.build_vocab(train_ds, max_size=10000, min_freq=3)
ENGLISH.build_vocab(train_ds, max_size=10000, min_freq=3)

print(f"Unique tokens in source (japanese) vocabulary: {len(JAPANESE.vocab)}")
print(f"Unique tokens in target (english) vocabulary: {len(ENGLISH.vocab)}")
print()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32

# dict_keys(['eng', 'jap'])
train_iterator, valid_iterator = BucketIterator.splits((train_ds, valid_ds),
                                                      batch_size = BATCH_SIZE,
                                                      sort_within_batch=True,
                                                      sort_key=lambda x: len(x.jap),
                                                      device = device)

input_size_encoder = len(JAPANESE.vocab)
embedding_size = 300
hidden_size = 1024
encoder_lstm = EncoderRNN(input_size_encoder, embedding_size,hidden_size).to(device)
#encoder_lstm = EncoderRNN(input_size_encoder, hidden_size).to(device)
print(encoder_lstm)
# EncoderRNN(
#   (embedding): Embedding(2144, 300)
#   (lstm): LSTM(300, 1024, num_layers=2, dropout=0.8)
#   (dropout): Dropout(p=0.8, inplace=False)
# )


input_size_decoder = len(ENGLISH.vocab)
output_size = len(ENGLISH.vocab)
decoder_lstm = DecoderRNN(input_size_decoder, embedding_size, hidden_size, output_size).to(device)
#decoder_lstm = DecoderRNN(hidden_size, output_size).to(device)
print(decoder_lstm)
# DecoderRNN(
#   (embedding): Embedding(1556, 1024)
#   (lstm): LSTM(300, 1024, num_layers=2)
#   (out): Linear(in_features=1024, out_features=1556, bias=True)
# )


######################################################################################################
class Seq2Seq(nn.Module):
    def __init__(self, Encoder_LSTM, Decoder_LSTM):
        super(Seq2Seq, self).__init__()
        self.Encoder_LSTM = Encoder_LSTM
        self.Decoder_LSTM = Decoder_LSTM

    def forward(self, source, target, tfr=0.5):
        # Shape - Source : (10, 32) [(Sentence length German + some padding), Number of Sentences]
        batch_size = source.shape[1]

        # Shape - Source : (14, 32) [(Sentence length English + some padding), Number of Sentences]
        target_len = target.shape[0]
        target_vocab_size = len(ENGLISH.vocab)

        # Shape --> outputs (14, 32, 5766)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        # Shape --> (hs, cs) (2, 32, 1024) ,(2, 32, 1024) [num_layers, batch_size size, hidden_size] (contains encoder's hs, cs - context vectors)
        hidden_state, cell_state = self.Encoder_LSTM(source)

        # Shape of x (32 elements)
        x = target[0]  # Trigger token <SOS>

        for i in range(1, target_len):
            # Shape --> output (32, 5766)
            output, hidden_state, cell_state = self.Decoder_LSTM(x, hidden_state, cell_state)
            outputs[i] = output
            best_guess = output.argmax(1)  # 0th dimension is batch size, 1st dimension is word embedding
            x = target[i] if random.random() < tfr else best_guess  # Either pass the next word correctly from the dataset or use the earlier predicted word

        # Shape --> outputs (14, 32, 5766)
        return outputs


learning_rate = 0.001
step = 0
model = Seq2Seq(encoder_lstm, decoder_lstm).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = ENGLISH.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
print(model)


def translate_sentence(model, sentence, japanese, english, device, max_length=50):
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_japanese(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens.insert(0, japanese.init_token)
    tokens.append(japanese.eos_token)
    text_to_indices = [japanese.vocab.stoi[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.Encoder_LSTM(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.Decoder_LSTM(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    return translated_sentence[1:]


num_epochs = 100
sentence1 = "Ich will dich eine Frage fragen"

for epoch in range(num_epochs):
    print("Epoch - {} / {}".format(epoch + 1, num_epochs))
    model.eval()
    translated_sentence1 = translate_sentence(model, sentence1, JAPANESE, ENGLISH, device, max_length=50)
    print(f"Translated example sentence 1: \n {translated_sentence1}\n")
    epoch_loss = 0.0

    model.train(True)
    for batch_idx, batch in enumerate(train_iterator):
        input = batch.jap.to(device)
        target = batch.eng.to(device)

        # Pass the input and target for model's forward method
        output = model(input, target)
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        # Clear the accumulating gradients
        optimizer.zero_grad()

        # Calculate the loss value for every epoch
        loss = criterion(output, target)

        # Calculate the gradients for weights & biases using back-propagation
        loss.backward()

        # Update the weights values using the gradients we calculated using bp
        optimizer.step()
        step += 1
        epoch_loss += loss.item()

    print(epoch_loss / len(train_iterator))