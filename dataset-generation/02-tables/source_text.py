from datasets import load_dataset
import time
import random

class TextSource:
    def get_characters(count):
        return "a"*count
    def get_words(count):
        return "a "*count

class ArxivSource(TextSource):
    def __init__(self):
        with open("/work/tesi_czaccagnino/small-custom/source.tex") as f:
            self.words = f.read().split()

        delchars = ''.join(c for c in map(chr, range(256)) if not (c.isalnum()))
        table = str.maketrans(delchars, len(delchars) * ' ')
        self.words = [el.translate(table) for el in self.words]

    def get_words(self, count):
        start_index = int((len(self.words)-count) * random.random())
        return " ".join(self.words[start_index:start_index+count])


class SentencesSource(TextSource):
    def __init__(self):
        dataset = load_dataset("generics_kb", split="train")
        dataset.shuffle(seed=time.time())
        self.sentences = dataset['generic_sentence']
        self.index = 0
        self.len = len(self.sentences)
        

    def get_characters(self, count):
        st = ""
        while len(st < count):
            if self.index == self.len:
                self.index = 0
            st += self.sentences.select(self.index) + " "
            self.index += 1
