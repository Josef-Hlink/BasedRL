# Policy Based Reinforcement Learning (pbrl)

This repo is called _Based RL_ because our agent is based, as opposed to being cringe.

Contributors:

- Josef Hamelink
- Koen van der Burg

## Setup

1. clone the repo
2. (recommended) create and activate virtual environment
   - `python3 -m venv venv`
   - `source venv/bin/activate`
3. build project
   - `pip install -e .`
4. install `torch` separately
   - `pip install torch` for normal version
   - `pip install torch --index-url https://download.pytorch.org/whl/cu118` if your machine has CUDA 11.8 or higher

## Usage

After building the project, the `pbrl` command should be available.
Run `pbrl --help` to see the available commands.
