from PIL import Image
from typing import Optional
from dataclasses import dataclass

import torch
from transformers.modeling_outputs import ModelOutput


@dataclass
class PegsOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_lm: Optional[torch.FloatTensor] = None
    loss_gen: Optional[torch.FloatTensor] = None


@dataclass
class PegsGenerationOutput(ModelOutput):
    output_sequence: Optional[torch.LongTensor] = None
    image: Optional[Image.Image] = None


@dataclass
class ConversationalAgentOutput(ModelOutput):
    text_response: Optional[str] = None
    image: Optional[Image.Image] = None