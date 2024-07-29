# zkML: Trade-offs in accuracy vs. proving cost

This repository contains the code for our blog post: https://www.hungrycats.studio/posts/zkml-tradeoffs/. 

In the land of (zk)SNARKS, building the most accurate machine learning model is often at odds with the costs of proving and verifying model inference.

To demonstrate the tradeoffs between model accuracy and SNARK costs, we’ve implemented a proof-of-concept using the EZKL zkML framework. Our goal is to highlight how small increases in accuracy might lead to significant computational expenses, encouraging thoughtful consideration of these tradeoffs when building models that need verifiability. This post provides a detailed explanation of the process, including data preprocessing, model training, and proof generation.

As proof-of-concept, we’ve chosen a Token Trend Forecasting task. This task involves binary classification, aiming to predict whether a token’s price will rise or fall in the future. zkML is particularly relevant for blockchain applications, and besides price prediction, it can also be useful for other critical tasks in the blockchain space, such as forecasting market volatility and assessing investment risk, as well as many non-financial applications. 

## Installation

The code requires Python 3.11 or later. To create a virtual environment with `venv` and install the required packages, run the following commands:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Running `pip freeze` should show `ezkl` and `giza` among the installed packages.

## Usage

All the code is contained in the `blog.ipynb` Jupyter notebook. It contains detailed explanations and code for data preprocessing, model training, and proof generation. Please refer to the blog post for a detailed explanation of the process.

## Acknowledgements

We would like to thank the authors of [EZKL](https://github.com/zkonduit) for granting us permission to utilize their library. Their comprehensive documentation and detailed tutorials made this project possible.