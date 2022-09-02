# ControlledCarp Implementation
> Fine tune language models with CARP

## Setup

1. Run `python setup.py develop`
2. `wget https://mystic.the-eye.eu/public/AI/CARP_L.pt` for pretrained CARP model
3. Install magiCARP repo
4. Install trl repo

## Train model

1. Put CARP model in magiCARP ckpts folder
2. Set config in ControlledCarp
3. Run python run.py --config PATH_TO_CONFIG
