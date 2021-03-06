# Development Environment Setup <!-- omit in toc -->

The goal of this document is to describe how to setup your development environment.

## Sections <!-- omit in toc -->

- [Python Setup](#python-setup)
  - [Install Python (v3.8)](#install-python-v38)
  - [Virtual Environment](#virtual-environment)
    - [Setup Python virtual environment via `VirtualEnv`](#setup-python-virtual-environment-via-virtualenv)
    - [Setup Python virtual environment via `PyEnv`](#setup-python-virtual-environment-via-pyenv)

## Python Setup

### Install Python (v3.8)

- Windows: <https://www.python.org/downloads/windows/>
  - **For x64 machines**, install `x86-64` version of Python.
  - Once Python is installed, add the following to your environment variables:

    | Variable      | Value                                            |
    | ------------- | ------------------------------------------------ |
    | %PYTHON_HOME% | {Folder path of your installed Python directory} |
    | Path          | %PYTHON_HOME%                                    |
    | Path          | %PYTHON_HOME%\Scripts                            |

  - Verify Python is properly installed with the following:

    ```bash
    $ python -V
    Python {version}
    ```

- Mac: it is recommend to use `pyenv` to setup and manage python:
  - `brew install pyenv`
  - `pyenv install {version}`
  - `pyenv global {version}`
  - Running `which python` should point to the .pyenv path e.g. `Users/USERNAME/.pyenv/shims/python`

Read more about [PyEnv](https://github.com/pyenv/pyenv).

### Virtual Environment

Do not run python commands without activating the Python virtual environment. You can use either method to setup a virtual environment,
pick the approach you prefer.

> You do *not* need both VirtualEnv and PyEnv.

Read more about [virtual environments](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/).

#### Setup Python virtual environment via `VirtualEnv`

This approach works on both Windows and Mac:

- Windows
  - Navigate inside your Python project directory.
  - `pip install virtualenv`
  - `python -m virtualenv venv`
  - To activate: `source venv/Scripts/activate`
    - If this doesn't work, try `. venv/Scripts/activate`
  - To deactivate: `deactivate`
- Mac or Linux
  - Navigate inside your Python project directory.
  - `pip install virtualenv`
  - `virtualenv venv`
  - `source venv/bin/activate`

Read more about [VirtualEnv](https://virtualenv.pypa.io/en/latest/).

#### Setup Python virtual environment via `PyEnv`

This approach is for Mac users:

- Install pyenv virtualenv with `brew install pyenv-virtualenv`
  - You need to have `pyenv` installed first
- Run `pyenv virtualenv`, specifying the Python version you want and the name of the virtualenv directory
  - E.g. `pyenv virtualenv 3.8.0 my-virtual-env-3.8.0`
- To activate the environment you can run `pyenv activate <name_of_virtualenv>`
- To deactivate `pyenv deactivate`

Read more about [PyEnv VirtualEnv Usage](https://github.com/pyenv/pyenv-virtualenv#usage).
