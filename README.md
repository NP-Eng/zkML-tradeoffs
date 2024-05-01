# Model Competition PoC

This repository contains a proof of concept for our blog post: https://www.hungrycats.studio/posts/zkml-tradeoffs/.

## Installation

To create a virtual environment with `venv` and install the required packages, run the following commands:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install the Giza stack and log in to the Giza server:

```
pipx install giza-cli --python $(which python) # Install the Giza CLI (on the .venv)
pip install giza-actions # Install the AI Actions SDK
```

## Usage

Log in to the Giza server:

```
giza users create # Create a user
giza users login # Login to your account
giza users create-api-key # Create an API key. We recommend you do this so you don't have to reconnect.
```
