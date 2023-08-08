from src.data.datahandler import Dataset
from src.engine.trainer import Trainer


class Launcher:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def run(self):
        # load summary dataset
        summary_dataset = Dataset(checkpoint=self.checkpoint)

        # get embeddings
        tokenized_data = summary_dataset.run()

        # fine-tune model
        trainer = Trainer(self.checkpoint)
        trainer.train(tokenized_data)
        

        

# inference
text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."

# The simplest way to try out your finetuned model for inference is to use it in a pipeline(). Instantiate a pipeline for summarization with your model, and pass your text to it:
from transformers import pipeline

summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
summarizer(text)


# You can also manually replicate the results of the pipeline if you'd like: Tokenize the text and return the input_ids as PyTorch tensors:
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
inputs = tokenizer(text, return_tensors="pt").input_ids


# Use the generate() method to create the summarization. For more details about the different text generation strategies and parameters for controlling generation, check out the Text Generation API.
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)


# Decode the generated token ids back into text
tokenizer.decode(outputs[0], skip_special_tokens=True)
