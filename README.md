# Description

Chat with PDF. Point the tool at a pdf and ask questions

# Usage

```bash
usage: main.py [-h] [-v] -i some pdf [-p prompt]

Chat with your pdf.

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -i some pdf, --input some pdf
                        path to the pdf
  -p prompt, --prompt prompt
                        optional custom prompt

```

## Example
'Fruits.pdf' file can be found in the folder named 'resources'. 

![Alt text](/resources/chat_pdf.gif)

# Requirements

My setup is:

WIN 10 + nvidia drivers 522 + cuda 11.8 + WSL2 with Ubuntu 22.04 with cuda tools + python >=3.10

see "pyproject.toml"

```bash
poetry install
```
