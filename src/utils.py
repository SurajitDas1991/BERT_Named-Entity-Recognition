import string
from torch import cuda
from transformers import BertTokenizer

class UtilsForProgram:
    device=string
    @classmethod
    def check_if_cuda_is_available(cls):
        cls.device='cuda' if cuda.is_available() else 'cpu'

    @staticmethod
    def tokenize_and_preserve_labels(sentence,text_labels,tokenizer:BertTokenizer):
        """
        Word piece tokenization makes it difficult to match word labels
        back up with individual word pieces. This function tokenizes each
        word one at a time so that it is easier to preserve the correct
        label for each subword.
        """
        tokenized_sentence = []
        labels = []

        sentence = sentence.strip()

        for word,label in zip(sentence.split(),text_labels.split(",")):
            #We tokenize the word and count # number of subwords it has been broken into
            tokenized_word=tokenizer.tokenize(word)
            n_subwords=len(tokenized_word)

            #Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)
            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label]*n_subwords)

        return tokenized_sentence,labels
