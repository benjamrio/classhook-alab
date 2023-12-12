# A-Lab : ClassHook

## Description

NLP-driven youtube video subject annotations.

Our model (finetuned BERT) computes the probability that a subtitle belongs to each of the chosen subjects. The user can also input a list of Youtube URLs, and our backend retrieves the subtitles from the Youtube API.

The models are pushed to huggingface hub, in a private repository.

## Preprocessing

The function to preprocess the data (merge, clean, select N subjects) is in `create_dataset.py`` and can be used as a standalone script:

```bash
python create_dataset.py --input_folder <path_to_input_folder> --output_folder <path_to_output_folder> --nb_subjects <nb_subjects>
```

## Training

Training is achieved on notebook `finetune.ipynb`. It can easily be executed on Colab using a T4 GPU. We recommend running training for between 5 and 7 epochs, and this takes less than an hour.

## Inference

Inference is easy to do. Two options:

1. Launch `inference.ipynb` on Colab. The server will be run on Colab's virtual machine. You can change the `remote` parameter if you want the server to be available online.

OR

2. launch the server from your computer. Execute these commands at the root of this folder.

```bash
pip install -r requirements
python webapp.py --remote <True/False>
```

The server runs on your  `http://127.0.0.1:7860/` if `remote` is `False` or at a distant link if `remote` is `True`.
