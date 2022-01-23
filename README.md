# ControlledCarp Implementation
> Fine tune language models with CARP

## Setup

1. Run `python setup.py develop`
2. `wget https://mystic.the-eye.eu/public/AI/CARP_L.pt` for pretrained CARP model

## Systemic Error Concerns

- GPTNeoHeadWithValueModel not properly loading all pretrianed weights
- - Same warning arises for GPT2
