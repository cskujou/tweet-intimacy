import torch
import pandas as pd
import datasets

from typing import Optional, Tuple, Union
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import XLMRobertaPreTrainedModel
from transformers import XLMRobertaModel
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import XLMRobertaTokenizerFast
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer

model_name = "cardiffnlp/twitter-xlm-roberta-base"
hidden_size = 256
num_layers = 2
bidirectional = True
dropout_gru = 0.1


class IntimacyNet(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config.problem_type = "regression"
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout_gru,
            batch_first=True,
        )
        self.reg = nn.Linear(hidden_size * 4, 1)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        states, _ = self.gru(self.dropout(outputs[0]))
        encoding = torch.cat((states[:, 0, :], states[:, -1, :]), dim=-1)
        logits = self.reg(self.dropout(encoding))

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    train_raw = pd.read_csv("train_cleaned.csv")
    test = pd.read_csv("semeval_test.csv")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in split.split(train_raw, train_raw["language"]):
        train = train_raw.iloc[train_idx]
        val = train_raw.iloc[val_idx]

    train_dict = {"label": train["label"], "text": train["text"]}
    val_dict = {"label": val["label"], "text": val["text"]}
    test_dict = {"text": test["text"]}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = IntimacyNet.from_pretrained(model_name, num_labels=1)
    metric = datasets.load_metric("pearsonr")

    training_args = TrainingArguments(
        output_dir="./checkpoint",  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        learning_rate=5e-6,
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="epoch",
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = prediction_outputs[0].flatten()
    result_output = pd.DataFrame(
        data={
            "text": test["text"],
            "language": test["language"],
            "prediction": test_pred,
        }
    )
    result_output.to_csv("XLM-T-GRU.csv", index=False)
    model.save_pretrained("xlm-t-gru")
