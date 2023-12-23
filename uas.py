import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Fungsi untuk mendapatkan frequent itemsets dan association rules
def get_apriori_results(transactions, min_support, min_confidence):
    # Membuat format data yang sesuai untuk algoritma Apriori
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Menerapkan algoritma Apriori untuk mendapatkan itemset yang sering muncul
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    # Menerapkan aturan asosiasi dari itemset yang sering muncul
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    return frequent_itemsets, rules

# Menampilkan tampilan aplikasi Streamlit
def main():
    st.title("Apriori Algorithm Demo")
 df = pd.read_csv('Groceries data.csv')
    # Contoh data transaksi
    transactions = [
        ['Roti', 'Susu', 'Telur'],
        ['Susu', 'Mentega'],
        ['Roti', 'Susu', 'Mentega'],
        ['Roti', 'Kopi'],
        ['Kopi']
    ]

    # Parameter untuk algoritma Apriori
    min_support = st.slider("Minimum Support", 0.0, 1.0, 0.2)
    min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.7)

    # Mendapatkan hasil dari algoritma Apriori
    frequent_itemsets, rules = get_apriori_results(transactions, min_support, min_confidence)

    # Menampilkan hasil
    st.subheader("Frequent Itemsets:")
    st.write(frequent_itemsets)

    st.subheader("Association Rules:")
    st.write(rules)

if __name__ == "__main__":
    main()
