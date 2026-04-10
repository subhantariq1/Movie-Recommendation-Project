import ast
import collections
import operator
import string

import nltk
import pandas as pd
import streamlit as st
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def ensure_nltk_data():
    resources = {
        "corpora/stopwords": "stopwords",
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }

    for path, pkg in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                # In some environments SSL blocks download.
                pass


def get_names(strlist):
    names = []
    for item in ast.literal_eval(strlist):
        names.append(item["name"])
    return names


def director(strlist):
    directors = []
    for item in ast.literal_eval(strlist):
        if item["job"] == "Director":
            directors.append(item["name"])
    return directors


def count_empty_lists(val):
    if isinstance(val, list):
        return 1 if not val else 0
    return 0


def no_punc(mylist):
    newlist = []
    for element in mylist:
        translator = str.maketrans("", "", string.punctuation)
        newlist.append(element.translate(translator))
    return newlist


def lowercase(mylist):
    return [element.lower() for element in mylist]


def remove_space(mylist):
    return [element.replace(" ", "") for element in mylist]


def extract_key_words(input_str, top_n_keywords=30):
    r = Rake()
    r.extract_keywords_from_text(input_str.lower())
    key_words_dict_scores = r.get_word_degrees()
    sorted_key_words_dict_scores = sorted(
        key_words_dict_scores.items(), key=operator.itemgetter(1), reverse=True
    )
    sorted_dict = collections.OrderedDict(sorted_key_words_dict_scores)
    return sorted_dict, list(sorted_dict.keys())[:top_n_keywords]


@st.cache_resource(show_spinner="Setting up recommendation engine. Please wait...")
def build_recommender():
    ensure_nltk_data()

    from nltk.corpus import stopwords
    from nltk.tokenize import RegexpTokenizer, word_tokenize

    movies_df_all = pd.read_csv("tmdb_5000_movies.csv")
    credits_df_all = pd.read_csv("tmdb_5000_credits.csv")

    movies_df = movies_df_all[
        [
            "title",
            "overview",
            "genres",
            "keywords",
            "production_companies",
            "release_date",
            "original_language",
        ]
    ].copy()
    credits_df = credits_df_all[["title", "cast", "crew"]].copy()

    outsource = pd.read_csv("imdb_top_1000.csv")[["Series_Title", "Overview"]].copy()
    outsource.rename(columns={"Series_Title": "title", "Overview": "overview"}, inplace=True)

    movies_df = pd.merge(
        movies_df, outsource, on="title", how="left", suffixes=("_main", "_missing")
    )
    movies_df["overview_main"] = movies_df["overview_main"].fillna(movies_df["overview_missing"])
    movies_df.drop(["overview_missing"], axis=1, inplace=True)
    movies_df.rename(columns={"overview_main": "overview"}, inplace=True)

    custom_overviews = {
        1990: "After the Rebels are overpowered by the Empire, Luke Skywalker begins his Jedi training with Yoda, while his friends are pursued across the galaxy by Darth Vader and bounty hunter Boba Fett.",
        2294: "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches and spirits, a world where humans are changed into beasts.",
        2656: "A biopic of the rise of father Jorge Mario Bergoglio SJ from a teacher in a Jesuit High School in Argentina to archbishop and cardinal of Buenos Aires to Pope of the Roman Catholic Church. The story touches on his relation with his fellow Jesuits in Argentina and Europe, to his relation with laureate writer Jorge Luis Borges, Argentine dictator Jorge Rafael Videla, and archbishops Laghi (nuncio to Argentina) and Quarracino (cardinal of Buenos Aires), up to the moment where he is elected Pope in 2013.",
        4140: "The life of Frank Sinatra, as an actor and singer and the steps along the way that led him to become such an icon.",
        4431: "There is so much interest in food these days yet there is almost no interest in the hands that pick that food. In the US, farm labor has always been one of the most difficult and poorly paid jobs and has relied on some of the nation's most vulnerable people. While the legal restrictions which kept people bound to farms, like slavery, have been abolished, exploitation still exists, ranging from wage theft to modern-day slavery. These days, this exploitation is perpetuated by the corporations at the top of the food chain: supermarkets. Their buying power has kept wages pitifully low and has created a scenario where desperately poor people are willing to put up with anything to keep their jobs.",
    }
    for idx, text in custom_overviews.items():
        movies_df.loc[idx, "overview"] = text

    movies_df.loc[movies_df.title == "America Is Still the Place", "release_date"] = "2022-06-10"

    clean1_df = pd.DataFrame(movies_df["title"])
    clean1_df["genres_list"] = movies_df["genres"].apply(get_names)
    clean1_df["keywords_list"] = movies_df["keywords"].apply(get_names)
    clean1_df["prod_companies_list"] = movies_df["production_companies"].apply(get_names)
    clean1_df["cast_list"] = credits_df["cast"].apply(get_names)
    clean1_df["director_list"] = credits_df["crew"].apply(director)

    def replace_empty_values(row):
        for col in clean1_df.columns:
            if len(row[col]) == 0:
                for other_col in clean1_df.columns:
                    if col != other_col and len(row[other_col]) > 0:
                        row[col] = row[other_col]
                        break
        return row

    _ = clean1_df.map(count_empty_lists).sum(axis=0)
    clean1_df = clean1_df.apply(replace_empty_values, axis=1)
    _ = clean1_df.map(count_empty_lists).sum(axis=0)

    clean1_df["overview_key_list"] = movies_df["overview"].apply(lambda x: extract_key_words(x)[1])
    clean1_df.iloc[:, 1:] = clean1_df.iloc[:, 1:].map(no_punc)
    clean1_df.iloc[:, 1:] = clean1_df.iloc[:, 1:].map(lowercase)

    def remove_stopwords(mylist):
        stop_words = set(stopwords.words("english"))
        newlist = []
        for element in mylist:
            words = word_tokenize(element)
            filtered_words = [word for word in words if word not in stop_words]
            newlist.append(" ".join(filtered_words))
        return newlist

    clean1_df["keywords_list"] = clean1_df["keywords_list"].apply(remove_stopwords)

    cols = ["genres_list", "keywords_list", "prod_companies_list", "cast_list", "director_list"]
    clean1_df[cols] = clean1_df[cols].map(remove_space)

    tokenizer = RegexpTokenizer(r"\w+")
    clean1_df["year"] = movies_df["release_date"].apply(lambda x: tokenizer.tokenize(x))
    for i in range(0, len(clean1_df)):
        del clean1_df["year"][i][1:3]

    wn = nltk.WordNetLemmatizer()

    def lemmatize(tokenized_text):
        return [wn.lemmatize(word) for word in tokenized_text]

    clean1_df["overview_key_list"] = clean1_df["overview_key_list"].apply(lemmatize)
    clean1_df["language"] = movies_df["original_language"].apply(lambda x: [x])

    df = clean1_df.drop("title", axis=1)
    df = df.map(lambda x: " ".join(x))
    merged_text = df.apply(lambda x: " ".join(x), axis=1)

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(merged_text)
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    clean1_df.set_index("title", inplace=True)
    movie_indices = pd.Series(clean1_df.index)
    movie_titles = sorted(clean1_df.index.tolist())

    return {
        "clean1_df": clean1_df,
        "movie_indices": movie_indices,
        "cosine_similarities": cosine_similarities,
        "movie_titles": movie_titles,
    }


def get_movie_recommendation(name, clean1_df, movie_indices, cosine_similarities):
    recommended_movies = []

    movie_index = movie_indices[movie_indices == name].index[0]
    score_series = pd.Series(cosine_similarities[movie_index]).sort_values(ascending=False)
    top_indexes = list(score_series.iloc[1:6].index)

    for i in top_indexes:
        recommended_movies.append(list(clean1_df.index)[i])

    return recommended_movies


st.title("Movie Recommender Engine")
st.write("Select a movie title to get 5 similar recommendations.")

artifacts = build_recommender()
selected_movie = st.selectbox("Choose a movie:", artifacts["movie_titles"])

if st.button("Recommend"):
    recommendations = get_movie_recommendation(
        selected_movie,
        artifacts["clean1_df"],
        artifacts["movie_indices"],
        artifacts["cosine_similarities"],
    )
    st.subheader("Recommended Movies")
    for i, movie in enumerate(recommendations, start=1):
        st.write(f"{i}. {movie}")