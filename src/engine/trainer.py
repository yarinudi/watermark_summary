import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from src.utils.logger import Logger


class Trainer:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.logger = Logger(log_path='resources\logs', name='trainer')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        self.rouge = evaluate.load("rouge")
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.training_args = self._set_trainer_args()

    def _set_trainer_args(self):      
        trainer_args = Seq2SeqTrainingArguments(
            output_dir="dong_model",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=4,
            predict_with_generate=True,
            # fp16=True,  # can also work on cude devices
            fp16=False,
            push_to_hub=False,
        )
        return trainer_args

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def train(self, tokenized_data):
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.checkpoint)
        
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
