from datasets import load_dataset
from dataclasses import dataclass
from transformers import AutoTokenizer

from src.data.preprocessor import preprocess_corpus


@dataclass
class Dataset:
      checkpoint: str =  "t5-small"
      data_dir: str = r"C:\Users\Yarin.Udi\PycharmProjects\watermark_summary\resources\data"
      prefix: str = "summarize: "
      name: str = "billsum"
      split: str = "ca_test"

      def __post_init__(self):
            self.data = load_dataset(self.name, split=self.split, data_dir=self.data_dir)

      def split_train_test(self):
            self.data = self.data.train_test_split(test_size=0.2)

            # display sample
            print(len(self.data["train"]), len(self.data["test"]), '\n', 
                  self.data['train'][0].keys(), self.data['train'][0])

      def preprocess(self):
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            tokenized_data = self.data.map(lambda batch: preprocess_corpus(batch, self.prefix, tokenizer), batched=True)

            return tokenized_data
      
      def run(self):
            self.split_train_test()

            tokenized_data = self.preprocess()

            return tokenized_data
