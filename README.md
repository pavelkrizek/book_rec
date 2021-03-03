# book_rec

Simple book recommendation engine with streamlit app. The app lets you choose the book from
the available books and returns the top n (you can control n) based on the similarity measure specified.
The recommendation is based on simple matrix factorization (this paper motivated this approach https://arxiv.org/abs/2005.09683), trained with Pytorch. Nearest Neighbors are found with annoy (https://github.com/spotify/annoy). Since it's sometimes difficult to judge whether
the recommendation makes sense at all, the book's covers, descriptions, and genre are also provided.

Hope it helps you to broaden your book horizons!

## How to run it

Install requirements
```bash
conda env create -f environment.yml
```

```bash
pip install -e .
```

To fetch Book-Crossing Dataset dataset, clean it and store it as parque file, run:
```bash
python book_rec/preprocessing.py
```

To train the model and store the book related embedding matrix, run:
```bash
python book_rec/train.py
```

Run the streamlit app
```bash
streamlit run book_rec/app.py
```

