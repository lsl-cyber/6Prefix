# 6Prefix

**6Prefix: A Routing-Prefix-Conditioned Transformer for IPv6 Target
Generation**


## Overview

The 6Prefix workflow consists of three main stages:

1.  **Preprocessing and encoding**\
    Routing-prefix--IPv6 address pairs are converted into fixed-length
    token sequences. Each routing prefix is represented by 16 two-digit
    hexadecimal fields plus its prefix length, forming a 17-token source
    sequence. Each IPv6 address is represented by 16 two-digit
    hexadecimal fields, forming a 16-token target sequence.

2.  **Routing-prefix-Conditioned Transformer training**\
    An encoder-decoder Transformer learns IPv6 address sequences
    conditioned on routing-prefix information. The implementation uses
    six encoder layers and six decoder layers, with a hidden dimension
    of 128, eight attention heads, a feed-forward dimension of 512, and
    dropout of 0.1.

3.  **Target generation with pre-probing**\
    For each target routing prefix, the generation script first performs
    small-scale candidate generation and multi-protocol pre-probing. The
    resulting pre-probing hit rates are used to allocate the formal
    generation budget across prefixes. Candidate addresses are then
    generated using Top-K sampling and dynamic temperature adjustment,
    followed by prefix validation and deduplication.

## Project Structure

``` text
6Prefix/
├── preprocess.py
├── train.py
├── generate_preprobe.py
├── probe.py
├── routing_prefix_ipv6.csv
├── src_sequences.npy
├── tgt_sequences.npy
├── token2id.json
├── id2token.json
├── transformer_ip_generator.pth
├── target_routing_prefixes.csv
└── ...
```


## Requirements

The implementation uses:

-   Python
-   PyTorch
-   NumPy
-   pandas

The generation script additionally uses Python's standard-library
networking modules for IPv6 validation and active probing, including
`ipaddress`, `socket`, `subprocess`, `platform`, and `concurrent.futures`.

A CUDA-capable GPU is recommended for model training. The training
script automatically uses CUDA when available and otherwise falls back
to CPU.

## 1. Data Preparation

The preprocessing script expects the following input file:

``` text
routing_prefix_ipv6.csv
```

The CSV file should contain at least the following columns:

``` text
routing_prefix
IPv6
```

The preprocessing script removes samples with missing routing prefixes
or IPv6 addresses.

The paper constructs the paired dataset by combining public IPv6 Hitlist
data with BGP routing tables and performing longest-prefix matching.
`preprocess.py` starts from the resulting paired dataset rather than
implementing the external BGP/pyasn data-collection procedure.

## 2. Preprocessing

Run:

``` bash
python preprocess.py
```

The script:

1.  Parses IPv6 routing prefixes and IPv6 addresses.
2.  Expands omitted IPv6 segments and pads hexadecimal fields.
3.  Splits each IPv6 address into 16 two-digit hexadecimal fields.
4.  Represents a routing prefix using 16 two-digit hexadecimal fields
    plus the prefix length.
5.  Builds a shared vocabulary containing hexadecimal fields,
    prefix-length tokens, `<PAD>`, and `<UNK>`.
6.  Encodes the source and target sequences.

The generated files are:

``` text
src_sequences.npy
tgt_sequences.npy
token2id.json
id2token.json
```

The resulting sequence dimensions are:

``` text
Source sequence: 17 tokens
Target sequence: 16 tokens
```

## 3. Model Training

Run:

``` bash
python train.py
```

The training script loads:

``` text
src_sequences.npy
tgt_sequences.npy
token2id.json
id2token.json
```

### Transformer configuration

The implementation uses:

``` text
Hidden dimension:       128
Attention heads:        8
Encoder layers:         6
Decoder layers:         6
Feed-forward dimension: 512
Dropout:                0.1
Batch size:             64
Epochs:                 100
Learning rate:          1e-4
Weight decay:           1e-5
LR scheduler:           CosineAnnealingLR
```

The model is optimized using AdamW and cross-entropy loss, with padding
tokens excluded from the loss calculation.

The best model according to the average training loss is saved as:

``` text
transformer_ip_generator.pth
```

## 4. Target Generation

The generation script requires:

``` text
target_routing_prefixes.csv
transformer_ip_generator.pth
token2id.json
id2token.json
```

The target prefix file contains one routing prefix per line, for example:

``` text
2001:4860:4860::/32
2400:3200::/12
```

Run:

``` bash
python generate_preprobe.py
```

For each target routing prefix, the script first generates a pre-probing
candidate set and probes candidate activity using:

``` text
ICMPv6
TCP/80
TCP/443
```

An address is treated as active when any enabled probing protocol
succeeds.


### Budget allocation

The formal generation budget is then allocated proportionally to the
pre-probing hit rates. A small smoothing term is used in the
implementation, and if all prefix hit rates are zero, the allocation
falls back to uniform allocation.

### Target Address generation

For each prefix, the generator:

1.  Encodes the routing prefix as a 17-token source sequence.
2.  Preserves the fixed prefix fields in the generated address.
3.  Autoregressively predicts the remaining address fields with the
    Transformer.
4.  Applies Top-K sampling.
5.  Uses a dynamic temperature that gradually decreases during repeated
    generation attempts.
6.  Checks whether each generated address belongs to the target routing
    prefix.
7.  Removes duplicate addresses.
8.  Allocates generation attempts according to the prefix-specific
    budget.

If the allocated budget cannot be fully generated, the implementation
performs additional top-up generation starting from prefixes with higher
pre-probing hit rates.

## 5. Generation Outputs

The generation script produces the following files:

``` text
preprobe_generated_ipv6.csv
preprobe_stats.csv
budget_allocation.csv
generated_ipv6.csv
```

The final output is deduplicated by IPv6 address and limited to the
configured total generation budget.

