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


if not isinstance(data, str):
    val_counts = df["itemDescription"].value_counts()
    product_count_pivot = val_counts.pivot_table(
        index='itemDescription', columns='Member_number', values='Count', aggfunc='sum').fillna(0)
    product_count_pivot = product_count_pivot.applymap(encode)

    frequent_itemsets_plus = apriori(product_count_pivot, min_support=0.03,
                                     use_colnames=True).sort_values('support', ascending=False).reset_index(drop=True)

    rules = association_rules(frequent_itemsets_plus, metric='lift',
                              min_threshold=1).sort_values('lift', ascending=False).reset_index(drop=True)[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules.sort_values('confidence', ascending=False, inplace=True)


def generate_candidates(prev_candidates, k):
    candidates = set()
    for i in range(len(prev_candidates)):
        for j in range(i + 1, len(prev_candidates)):
            itemset1 = set(prev_candidates[i])
            itemset2 = set(prev_candidates[j])
            union_set = itemset1.union(itemset2)
            if len(union_set) == k:
                candidates.add(tuple(sorted(union_set)))
    return list(candidates)

def prune_candidates(candidates, prev_frequent_sets):
    pruned_candidates = []
    for candidate in candidates:
        subsets = [set(x) for x in itertools.combinations(candidate, len(candidate) - 1)]
        is_valid = all(subset in prev_frequent_sets for subset in subsets)
        if is_valid:
            pruned_candidates.append(candidate)
    return pruned_candidates

def apriori(transactions, min_support):
    itemsets = [frozenset([item]) for item in set(item for transaction in transactions for item in transaction)]
    frequent_itemsets = []
    
    k = 2
    while itemsets:
        candidates = generate_candidates(itemsets, k)
        item_counts = {candidate: 0 for candidate in candidates}
        
        for transaction in transactions:
            for candidate in candidates:
                if set(candidate).issubset(transaction):
                    item_counts[candidate] += 1

        frequent_itemsets_k = [itemset for itemset, count in item_counts.items() if count / len(transactions) >= min_support]
        frequent_itemsets.extend(frequent_itemsets_k)
        
        itemsets = prune_candidates(generate_candidates(frequent_itemsets_k, k+1), frequent_itemsets_k)
        k += 1
    
    return frequent_itemsets

# Contoh Penggunaan
transactions = [
    ['whole milk', 'sausage', 'curd'],
]

min_support = 0.2
result = apriori(transactions, min_support)
print("Frequent Itemsets:")
print(result)




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
