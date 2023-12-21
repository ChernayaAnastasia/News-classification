from typing import Dict

import torch
from transformers import AutoModel

class ModelForClassification(torch.nn.Module):

    def __init__(self, model_path: str, config: Dict, custom_config: Dict):
        super(ModelForClassification, self).__init__()
        self.model_name = model_path
        self.config = config
        self.custom_config = custom_config
        self.n_classes = custom_config['num_classes']
        self.dropout_rate = custom_config['dropout_rate']
        self.bert = AutoModel.from_pretrained(self.model_name, output_hidden_states=True)
        self.pre_classifier = torch.nn.Linear(self.config.hidden_size*4, 768)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.classifier = torch.nn.Linear(768, self.n_classes)
        self.softmax = torch.nn.LogSoftmax(dim = 1)

    def forward(self, input_ids, attention_mask,):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        all_hidden_states = torch.stack(output[2])
        concatenate_pooling = torch.cat((all_hidden_states[-1],
                                         all_hidden_states[-2],
                                         all_hidden_states[-3],
                                        all_hidden_states[-4]),-1)
        concatenate_pooling = concatenate_pooling[:, 0]
        logits = self.pre_classifier(concatenate_pooling) # regression head
        relu_state = torch.nn.ReLU()(logits)
        dropout_state = self.dropout(relu_state)
        output = self.classifier(dropout_state)
        output = self.softmax(output)
        return output