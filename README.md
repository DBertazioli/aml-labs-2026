# Advanced Machine Learning — lab sessions (UNIMIB, A.Y. 2026/27)

Notebooks for the lab sessions of Advanced Machine Learning. Everything runs in Google Colab; a GPU is optional.

| | Notebook | |
|---|---|---|
| Before the labs | PyTorch warm-up + class server connection test | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DBertazioli/aml-labs-2026/blob/main/labs/00_prelab/prelab_pytorch.ipynb) |
| | Solutions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DBertazioli/aml-labs-2026/blob/main/labs/00_prelab/prelab_pytorch_solutions.ipynb) |

The warm-up is optional study, but its last section (connecting to the class server) is required before the first lab on 27 October.

## Class server

During the labs you send results to a class server, so we can build plots and leaderboards together. In Colab:

```python
!wget -q -O aml_client.py https://raw.githubusercontent.com/DBertazioli/aml-labs-2026/main/client/aml_client.py
import aml_client as aml
aml.register("name.surname@campus.unimib.it", "your_nickname")
```

Your campus email is used only to contact competition winners. It is never shown and is deleted after the course. Your nickname appears on the projector, so do not use your real name.
