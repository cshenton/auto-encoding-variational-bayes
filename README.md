# Variational Auto-Encoder (vanilla)
Replication of [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf) (Kingma &amp; Welling, 2013)


## Quick Start

```bash
# Create and activate virtual environment
virtualenv -p python3.5 venv
source venv/bin/activate

# Install dependencies with pip
pip install -r requirements.txt

# Run main.py, which trains vae and saves results to /img
python main.py
```

## More Details

The variational autoencoder implementation is in vanilla tensorflow, and is in `/vae`.
Since the same graph can be used in multiple ways, there is a simple `VAE` class that
constructs the `tf` graph and has useful pointers to important tensors and methods to
simplify interaction with those tensors.


## Thoughts

- VAE for MNIST
- VAE for Frey Face
- Functions to make encoders, decoders
- Simple object for full graph
    - inference method for image sim
    - accessible input and loss
- Make Figure 2 for z dim = 10

Chuck all that stuff in package vae.

Then in main.py in root, load those and relevant graphing tools, train the model, make the graphs and images.
Then save to some gitignored subfolder. So our tf code is nicely separated but we can still easily go

```bash
python main.py
```

to generate some images.