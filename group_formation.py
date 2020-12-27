import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from ast import literal_eval
import data_preprocessing as dp


def plot_group_similarity(group_sim, file):
    plt.figure()
    plt.hist([[np.round(sim, 3) for clique, sim in group_sim if len(clique) == 3],
              [np.round(sim, 3) for clique, sim in group_sim if len(clique) == 4],
              [np.round(sim, 3) for clique, sim in group_sim if len(clique) == 5]],
             density=True, bins=20,  edgecolor='black', linewidth=0.5,
             label=["Group Size: 3", "Group Size: 4", "Group Size: 5"])  # `density=False` would make counts
    plt.ylabel('Pecentage of Groups')
    plt.xlabel('Cosine Similarity')
    plt.legend(loc='upper right')
    plt.savefig(file + "_group_similarity.pdf", bbox_inches='tight', dpi=900)


def group_similarity(cliques_list, df_reviews):
    group_sim = []
    for clique in cliques_list:
        reviews_temp = df_reviews[df_reviews["user_id"].isin(clique)]
        reviews_temp = reviews_temp.groupby(["user_id", "business_id"])["stars"].mean().reset_index()
        reviews_matrix = reviews_temp.pivot(index='user_id', columns='business_id', values='stars')
        reviews_matrix = reviews_matrix.replace(np.nan, 0)
        mat_sparse = sparse.csr_matrix(reviews_matrix.values)
        similarities = cosine_similarity(mat_sparse, dense_output=True)
        sim = np.triu(similarities, 1).sum()
        if sim > 0:
            sim = sim/np.count_nonzero(np.triu(similarities, 1))
        if sim > 0:
            group_sim.append((clique, sim))
    return group_sim


def get_business_count(group, reviews):
    reviews = reviews[reviews["user_id"].isin(group)]
    reviews = reviews.groupby("user_id")["business_id"].apply(list).to_frame().reset_index()
    intersection = set.intersection(*map(set, reviews["business_id"].tolist()))
    return len(intersection)


def compute_groups(city, review_th, min_csize, max_csize, c_th, min_gsize, max_gsize):
    df_user, df_business, df_reviews = dp.get_city_data(city, review_th)

    business = df_reviews.groupby("business_id")["user_id"].apply(list).to_frame().reset_index()
    business = business[business["user_id"].apply(lambda x: len(x) >= 20)]
    business_list = business["business_id"].tolist()
    user_list = business["user_id"].tolist()
    user_list = list(set([subitem for item in user_list for subitem in item]))

    df_reviews = df_reviews[df_reviews["business_id"].isin(business_list)]

    df_user["friends_city"] = df_user["friends"].apply(lambda x: [f for f in x if f in user_list])
    df_user = df_user[df_user["friends_city"].apply(lambda x: len(x) > 0)]
    edge_list = df_user[["user_id", "friends_city"]].apply(lambda x: [(x[0], f) for f in x[1]], axis=1).tolist()
    edge_list = [subitem for item in edge_list for subitem in item]
    edge_list = [tuple(sorted(item)) for item in edge_list]
    edge_list = list(set(edge_list))

    G = nx.Graph()
    G.add_edges_from(edge_list)
    cliques_list = list(nx.find_cliques(G))
    cliques_list = [clique for clique in cliques_list if len(clique) >= min_csize and len(clique) <= max_csize]

    cliques_list = group_similarity(cliques_list, df_reviews.copy(deep=True))
    cliques_lt = [clique for clique, sim in cliques_list if sim >= c_th]

    subcliques_list = []
    for clique in cliques_lt:
        for r in range(min_gsize, max_gsize + 1):
            subcliques_list += list(combinations(clique, r))

    subcliques_list = list(set(subcliques_list))
    subcliques_list = group_similarity(subcliques_list, df_reviews.copy(deep=True))

    df_group = pd.DataFrame(subcliques_list, columns=["group", "similarity"])
    df_group["business_count"] = df_group["group"].apply(lambda x: get_business_count(x, df_reviews.copy(deep=True)))
    df_group = df_group[df_group["business_count"] > 0]
    df_group.to_csv(city[0] + "_groups_test.csv", header=True, index=False)


def get_groups(city, cr_count, threshold):
    df_group = pd.read_csv(city[0] + "_groups.csv")
    df_group["group"] = df_group["group"].apply(lambda x: literal_eval(x))
    df_group["group_id"] = df_group["group"].apply(lambda x: "-".join(sorted(x)))
    df_group = df_group.drop_duplicates(subset=["group_id"]).drop("group_id", axis=1)

    # Filter groups based on co-ratings
    df_group = df_group[df_group["business_count"] >= cr_count]
    df_group["group_size"] = df_group["group"].apply(lambda x: len(x))

    # Remove outliers with very high or low group similarity
    q_low = df_group["similarity"].quantile(0.0005)
    q_hi = df_group["similarity"].quantile(0.9995)
    t = len(df_group)
    df_group = df_group[(df_group["similarity"] < q_hi) & (df_group["similarity"] > q_low)]
    print("No of outliers removed for : ", (t-len(df_group)))

    # Sample groups
    df_group = df_group.sample(frac=threshold, replace=False, random_state=1)

    # Plot group similarity
    plot_group_similarity([tuple(x) for x in df_group[["group", "similarity"]].values], city[0])
    print("Mean Group Similarity:", np.mean(df_group["similarity"].tolist()))
    print("Std Group Similarity:", np.std(df_group["similarity"].tolist()))
    print("Mean Group Business Count:", np.mean(df_group["business_count"].tolist()))
    print("Std Group Business Count:", np.std(df_group["business_count"].tolist()))

    group_list = df_group[["group", "similarity"]].values.tolist()
    return group_list


def get_business_list(group, df_reviews):
    reviews = df_reviews[df_reviews["user_id"].isin(group)]
    reviews = reviews.groupby("user_id")["business_id"].apply(list).to_frame().reset_index()
    intersection = list(set.intersection(*map(set, reviews["business_id"].tolist())))
    return intersection


def split_train_test(city, review_th, group_list, test_size):
    # Get reviews for the given city
    _, _, df_reviews = dp.get_city_data(city, review_th)

    df_group = pd.DataFrame(group_list, columns=["group", "similarity"])

    # Get restaurants visited by each group
    df_group["business_list"] = df_group["group"].apply(lambda x: get_business_list(x, df_reviews.copy(deep=True)))
    df_group["business_list"] = df_group[["group", "business_list"]].apply(lambda x: [(x[0], item) for item in x[1]], axis=1)

    data = [pair for group in df_group["business_list"].tolist() for pair in group]
    data = pd.DataFrame(data, columns=["group", "item"])
    train = pd.DataFrame()
    test = pd.DataFrame()
    # For each group split train and test data
    for group in data["group"].unique().tolist():
        gdata = data[data["group"] == group].drop_duplicates()
        # Choose the test samples randomly
        gtest = gdata.sample(n=test_size, replace=False, random_state=1)
        test = pd.concat([test, gtest], sort=True)
    train = data[~(data.index.isin(test.index))]

    # Train and test set for groups
    group_test = test.values.tolist()
    group_train = train.values.tolist()

    return group_train, group_test
