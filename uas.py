import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("Groceries data.csv")

st.title("Market Analyst")


def get_data(itemDescription='', year=''):
    data = df.copy()
    filtered = data.loc[
        (data["itemDescription"].str.contains(itemDescription)) &
        (data["year"].astype(str).str.contains(year))
    ]
    return filtered if not filtered.empty else "No Result"


def user_input_features():
    Product = st.selectbox("Member_number", ['1808', '2552', '2300', '1187',
                           '3037', '4941', '4501'])
    itemDescription = st.selectbox("itemDescription", ['tropical fruit', 'whole milk', 'pip fruit',
                                   'other vegetables', 'whole milk', 'rolls/buns', 'other vegetables', 'pot plants', 'whole milk'])
    year = st.select_slider("year", list(map(str, range(1, 42))))
    return itemDescription, year, Product


itemDescription, year, Product = user_input_features()

data = get_data(itemDescription.lower(), year)


def encode(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1



def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)


def return_product_df(product_antecedents):
    data = rules[["antecedents", "consequents"]].copy()

    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)

    return list(data.loc[data["antecedents"] == product_antecedents].iloc[0, :])


if type(data) != type("No Result"):
    st.markdown("Rekomendasi: ")
    st.success(
        f"Jika konsumen membeli **{product}**, maka membeli **{return_product_df(product)[1]}** secara bersamaan")
