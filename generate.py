from train import Model
from train import Preprocessor
from train import Classifier
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text Generation')
    parser.add_argument('--model', type=str, help='path to model')
    parser.add_argument('--prefix', type=str, default=None, help='input to model')
    parser.add_argument('--length', type=int, default=5, help='length of generated text')
    arguments = parser.parse_args()

    preprocessor = Preprocessor(mode="test")

    if arguments.prefix:
        preprocessed_text = preprocessor.preprocess([arguments.prefix])[0]
    model = Model.load_model(f'{arguments.model}')
    print(' '.join(model.generate(preprocessed_text, seq_len=arguments.length)))
