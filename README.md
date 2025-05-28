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

## Usage

After building the project, the following commands are available:

- `pbrl-run` to run a single experiment
- `pbrl-render` to render the results of a single experiment (if it was saved)
- `pbrl-sweep` to run a sweep of experiments using [wandb](https://wandb.ai/)

To see what arguments each command takes, run the command with either the `-h` or `--help` flag.

>[!CAUTION]
> In order to do any rendering (`pbrl-render`) you'll need [tkinter](https://docs.python.org/3/library/tkinter.html).

```bash
sudo apt install python3-tk  # on ubuntu/deb with apt
brew install python3-tk      # on macos with brew
```
