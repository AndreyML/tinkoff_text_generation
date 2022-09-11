import torch
from torch import nn
from gensim.utils import tokenize
import argparse
import pickle
import random
from typing import Any, List, Tuple, Union
import os
import numpy as np
import torch
from torch import nn
from gensim.models import Word2Vec
from tqdm import tqdm


class Preprocessor:
    def __init__(self, mode: str = "test"):
        self.mode = mode.lower()

    def tokenize(self, sentence: str) -> list:
        tokenized_sentence = list(tokenize(sentence, lowercase=True, deacc=True))
        return tokenized_sentence

    def add_special_tokens(self, sentence: list) -> list:
        sentence.insert(0, "pad")
        sentence.insert(1, 'bos')
        if self.mode == "train":
            sentence.insert(len(sentence), 'eos')

        return sentence

    def preprocess(self, text: list) -> list:
        text = list(map(self.tokenize, text))
        text = list(sen for sen in text if sen)
        text = list(map(self.add_special_tokens, text))

        return text


class Classifier(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, last_n: int, vocab_size: int) -> torch.tensor:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.last_n = last_n
        self.vocab_size = vocab_size

        self.context_linear = nn.Linear(embedding_dim, hidden_dim)
        self.last_n_linears = [nn.Linear(embedding_dim, hidden_dim)] * self.last_n
        self.global_linear = nn.Linear(hidden_dim * (self.last_n + 1), hidden_dim)
        self.classifier = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, context, last_n_embeddings) -> torch.tensor:
        context = context.float()
        last_n_embeddings = last_n_embeddings.float()

        x_context = [self.context_linear(context)]
        x_last_n = [self.last_n_linears[i](last_n_embeddings[i]) for i in range(self.last_n)]

        x = self.global_linear(torch.cat(x_context + x_last_n))
        probabilities = self.classifier(x)

        return probabilities


class Model(nn.Module):
    def __init__(self, word2vec: Word2Vec = None, loss_function: nn.Module = None, optimizer: Any = None,
                 lr: float = 3e-4, device: torch.device = None, vocab: dict = None, vocab_size: int = None,
                 embedding_dim: int = None, hidden_dim: int = None, last_n: int = None):
        super().__init__()
        self.new_state = None
        self.word2vec: Word2Vec = word2vec

        self.loss_function = loss_function
        self.lr = lr
        self.device = device

        self.classifier = Classifier(embedding_dim, hidden_dim, last_n, vocab_size)
        self.optimizer = optimizer(self.classifier.parameters(), lr=self.lr)

        self.vocab = vocab
        self.invert_vocab = {v: k for k, v in vocab.items()}

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.last_n = last_n

        self.context: np.ndarray = np.zeros(self.embedding_dim)
        self.last_n_embeddings: np.ndarray = np.zeros((self.last_n, self.embedding_dim))
        self.current_seq_len: int = 0

    def get_embeddings(self, tokens: list) -> Tuple[torch.tensor, torch.tensor]:
        embeddings = self.word2vec.wv[tokens]
        self.last_n_embeddings = np.vstack([self.last_n_embeddings, embeddings])[-2:]
        self.new_state = np.mean(embeddings, axis=0)

        new_seq_len = len(tokens)
        whole_seq_len = self.current_seq_len + new_seq_len

        last_proportion = self.current_seq_len / whole_seq_len
        new_proportion = new_seq_len / whole_seq_len

        self.context = last_proportion * self.context + new_proportion * self.new_state

        return torch.from_numpy(self.context), torch.from_numpy(self.last_n_embeddings)

    def reset(self):
        self.context: np.ndarray = np.zeros(self.embedding_dim)
        self.last_n_embeddings: np.ndarray = np.zeros((self.last_n, self.embedding_dim))
        self.current_seq_len: int = 0

    def train_epoch(self, model, optimizer, loss_function: object, device, vocab, lines):
        model.to(device)
        model.train()

        epoch_loss = 0.0
        lines_len = len(lines)
        vocab_size = len(list(vocab.keys()))

        for line in tqdm(lines):
            seq_len = len(line)
            split_index = seq_len // 2
            input = line[:split_index]

            sentence_loss = 0.0
            for i in range(split_index, seq_len - 1):
                context, last_n_embeddings = self.get_embeddings(input)
                context, last_n_embeddings = context.to(device), last_n_embeddings.to(
                    device
                )
                output = model(context, last_n_embeddings)

                target = torch.zeros(vocab_size)
                target_id = torch.tensor([vocab.get(line[i])])
                target[target_id] = 1
                target, output = target.unsqueeze(0), output.unsqueeze(0)
                loss = loss_function(output, target)
                sentence_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            sentence_loss /= seq_len // 2 - 1
            epoch_loss += sentence_loss
            self.reset()

        epoch_loss /= lines_len
        return epoch_loss

    def eval_epoch(self, model, loss_function, device, vocab, lines):
        model.to(device)
        model.eval()

        lines_len = len(lines)
        epoch_loss = 0.0
        vocab_size = len(list(vocab.keys()))
        for line in tqdm(lines):
            seq_len = len(line)
            split_index = seq_len // 2
            input = line[:split_index]

            sentence_loss = 0.0
            for i in range(split_index, seq_len - 1):
                context, last_n_embeddings = self.get_embeddings(input)
                context, last_n_embeddings = context.to(device), last_n_embeddings.to(
                    device
                )
                with torch.no_grad():
                    output = model(context, last_n_embeddings)

                target = torch.zeros(vocab_size)
                target_id = torch.tensor([vocab.get(line[i])])
                target[target_id] = 1
                target, output = target.unsqueeze(0), output.unsqueeze(0)
                loss = loss_function(output, target)
                sentence_loss += loss.item()

            sentence_loss /= seq_len // 2 - 1
            epoch_loss += sentence_loss
            self.reset()

        epoch_loss /= lines_len
        return epoch_loss

    def fit(self, text, eval_percent: float = 0.2):
        total_examples = len(text)
        split_int = int(total_examples * (1 - eval_percent))

        train_loss = self.train_epoch(
            model=self.classifier,
            lines=text[:split_int],
            loss_function=self.loss_function,
            device=self.device,
            vocab=self.vocab,
            optimizer=self.optimizer
        )
        print("Train Loss:", train_loss)

        eval_loss = self.eval_epoch(
            model=self.classifier,
            lines=text[split_int:],
            loss_function=self.loss_function,
            device=self.device,
            vocab=self.vocab,
        )

        print("Eval Loss:", eval_loss)

    def generate(self, tokens, seq_len) -> str:
        if tokens is None:
            tokens = ['pad', 'bos']
            tokens += [self.invert_vocab[random.randint(0, self.vocab_size - 1)]]

            context, last_n_embeddings = self.get_embdeddings(tokens)
            for i in range(seq_len):
                probabilities = self.classifier(context, last_n_embeddings).softmax(0)
                probabilities, indices = torch.sort(probabilities)
                j = -1
                id = indices[j].item()
                word = self.invert_vocab[id]
                while word in tokens:
                    j -= 1
                    id = indices[j].items()
                    word = self.invert_vocab[id]

                tokens += [word]
                context, last_n_embeddings = self.get_embeddings([word])
            return ''.join(tokens)
        else:

            context, last_n_embeddings = self.get_embeddings(np.reshape(tokens, -1))
            tokens = list(np.reshape(tokens, -1))
            for i in range(seq_len):
                ids = torch.topk(self.classifier(context, last_n_embeddings), 20).indices.numpy()
                id = np.random.choice(ids)
                word = self.invert_vocab[id]
                tokens += [word]
                context, last_n_embeddings = self.get_embeddings([word])
            return tokens[2:]

    def save_model(self, path: str) -> None:
        if not path.endswith("pkl"):
            raise ValueError('Model extension must be .pkl')

        with open(f'{path}', 'wb') as f:
            pickle.dump(self, file=f)

    @staticmethod
    def load_model(path: str) -> "Model":
        if not path.endswith("pkl"):
            raise ValueError("Model extension must be .pkl")

        with open(path, "rb") as f:
            model = pickle.load(file=f)

        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--input-dir', type=str, help='Data directory path')
    parser.add_argument('--model', type=str, help='Model save path')
    arguments = parser.parse_args()
    VECTOR_SIZE = 32
    preprocessor = Preprocessor(mode="train")
    if arguments.input_dir:
        for file in os.listdir(arguments.input_dir):
            with open(f"{arguments.input_dir}/{file}", "r") as f:
                text = f.readlines()
                preprocessed_text = preprocessor.preprocess(text)[:15000]
    else:
        text = input().split("\n")
        preprocessed_text = preprocessor.preprocess(text)[:15000]

    word2vec = Word2Vec(sentences=preprocessed_text, vector_size=VECTOR_SIZE, min_count=1, sg=1)
    word2vec.train(preprocessed_text, total_examples=len(preprocessed_text), epochs=20)

    vocab = word2vec.wv.key_to_index
    vocab_size = len(list(vocab.keys()))

    model = Model(
        word2vec=word2vec,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        lr=0.01,
        device=torch.device("cpu"),
        embedding_dim=VECTOR_SIZE,
        hidden_dim=VECTOR_SIZE,
        last_n=2,
        vocab=vocab,
        vocab_size=vocab_size,
    )

    model.fit(preprocessed_text)

    model.save_model(arguments.model)
