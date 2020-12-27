import pandas as pd
import data_preprocessing as dp
import re
from collections import defaultdict
from surprise import Dataset
from surprise import Reader
from surprise import SVD, SVDpp, NMF
from surprise import accuracy


def get_profile(group, business_list, reviews):
    reviews = reviews[reviews["user_id"].isin(group)]
    reviews = reviews[reviews["business_id"].isin(business_list)]
    reviews = reviews.groupby("business_id")["stars"].mean().to_frame().reset_index().values.tolist()
    return reviews


def group_rating(group, business_id, reviews):
    reviews = reviews[reviews["user_id"].isin(group)]
    reviews = reviews[reviews["business_id"] == business_id]
    rating = reviews["stars"].mean()
    return rating


def build_group_profile(city, review_th, group_partition, group_train):
    _, _, df_reviews = dp.get_city_data(city, review_th)
    group_partition["valid_business"] = group_partition["valid_business"].apply(lambda x: [i for i in re.split("[],;[ '()]", x) if len(i) > 0])
    group_partition["group"] = group_partition["group"].apply(lambda x: [i for i in re.split("[],;[ '()]", x) if len(i) > 0])
    group_partition["id"] = group_partition["group"].apply(lambda x: "-".join(sorted(x)))
    group_partition = group_partition.drop(["partition_info", "group"], axis=1)
    group_train = pd.DataFrame(group_train, columns=["group", "business_id"])
    group_train["id"] = group_train["group"].apply(lambda x: "-".join(sorted(x)))
    df_train = group_train.merge(group_partition, on="id", how="left")
    df_train = df_train.dropna()
    df_train["check"] = df_train[["valid_business", "business_id"]].apply(lambda x: 1 if x[1] in x[0] else 0, axis=1)
    df_train = df_train[df_train["check"] == 1]
    df_train["rating"] = df_train[["group", "business_id"]].apply(lambda x: group_rating(x[0], x[1], df_reviews.copy(deep=True)), axis=1)
    df_train = df_train[["group_id", "business_id", "rating"]]
    return df_train


def get_test_data(city, review_th, group_partition, group_test):
    _, _, df_reviews = dp.get_city_data(city, review_th)
    df_test = pd.DataFrame(group_test, columns=["group", "business_id"])
    df_test["rating"] = df_test[["group", "business_id"]].apply(lambda x: group_rating(x[0], x[1], df_reviews.copy(deep=True)), axis=1)
    group_partition["group"] = group_partition["group"].apply(lambda x: [i for i in re.split("[],;[ '()]", x) if len(i) > 0])
    group_partition["group"] = group_partition["group"].apply(lambda x: "-".join(sorted(x)))
    df_test["group"] = df_test["group"].apply(lambda x: "-".join(sorted(x)))
    df_test = df_test.merge(group_partition[["group", "group_id"]], on="group", how="left")
    df_test = df_test.dropna()
    return df_test[["group_id", "business_id", "rating"]]


def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def get_top_n_partition(predictions, group_partition, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    top_n_new = defaultdict(list)
    for uid, user_ratings in top_n.items():
        valid = group_partition[group_partition["group_id"] == uid]["valid_business"].to_list()[0]
        temp = []
        for iid, rating in user_ratings:
            if iid in valid:
                temp.append((iid, rating))
        temp.sort(key=lambda x: x[1], reverse=True)
        top_n_new[uid] = temp[:n]
    return top_n_new


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


def hit_ratio(top_n, N, test_data):
    count = 0
    arhr = 0
    # Print the recommended items for each user
    for uid, user_ratings in top_n.items():
        pred = []
        for iid, _ in user_ratings:
            pred.append(iid)
        if uid in test_data["group_id"].unique().tolist():
            test = test_data[test_data["group_id"] == uid]
            test = test["business_id"].values.tolist()[0]
            if test in pred:
                count += 1
                arhr += 1/(pred.index(test)+1)

    hr = count/test_data["group_id"].nunique()
    arhr = arhr/test_data["group_id"].nunique()
    return hr, arhr


def group_recommendation(group_profile, test_dataset, group_partition, algo_type, N=10):
    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 5))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(group_profile[['group_id', 'business_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    testset = Dataset.load_from_df(test_dataset[['group_id', 'business_id', 'rating']], reader)

    if algo_type == "svd":
        algo = SVD()
    if algo_type == "svdpp":
        algo = SVDpp()
    if algo_type == "nmf":
        algo = NMF()
    algo.fit(trainset)

    # Train the algorithm on the trainset, and predict ratings for the testset
    predictions = algo.test(test_dataset.values.tolist())
    # RMSE & MAE Accuracy
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    # Than predict ratings for all pairs (u, i) that are NOT in the training set.
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)

    # Evaluate Top-N Reccomendations
    top_n = get_top_n_partition(predictions, group_partition, n=N)
    hr, arhr = hit_ratio(top_n, N, test_dataset)
    return rmse, mae, hr, arhr


def get_baseline_data(city, review_th, group_train, group_test):
    _, _, df_reviews = dp.get_city_data(city, review_th)
    df_train = pd.DataFrame(group_train, columns=["group", "business_id"])
    df_train["rating"] = df_train[["group", "business_id"]].apply(lambda x: group_rating(x[0], x[1], df_reviews.copy(deep=True)), axis=1)
    df_train["temp"] = df_train["group"].apply(lambda x: "-".join(sorted(x)))

    df_test = pd.DataFrame(group_test, columns=["group", "business_id"])
    df_test["rating"] = df_test[["group", "business_id"]].apply(lambda x: group_rating(x[0], x[1], df_reviews.copy(deep=True)), axis=1)
    df_test["temp"] = df_test["group"].apply(lambda x: "-".join(sorted(x)))

    df = df_test[["temp"]].drop_duplicates()
    df["group_id"] = df.index

    df_train = df_train.merge(df, on="temp", how="left")
    df_train = df_train[["group_id", "business_id", "rating"]]
    df_train = df_train.dropna()

    df_test = df_test.merge(df, on="temp", how="left")
    df_test = df_test[["group_id", "business_id", "rating"]]
    df_test = df_test.dropna()
    return df_train, df_test


def baseline_recommendation(train_data, test_data, algo_type, N=10):
    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 5))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(train_data[['group_id', 'business_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    testset = Dataset.load_from_df(test_data[['group_id', 'business_id', 'rating']], reader)

    if algo_type == "svd":
        algo = SVD(n_factors=30)
    if algo_type == "svdpp":
        algo = SVDpp()
    if algo_type == "nmf":
        algo = NMF()
    algo.fit(trainset)

    # Train the algorithm on the trainset, and predict ratings for the testset
    predictions = algo.test(test_data.values.tolist())
    # RMSE & MAE Accuracy
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    # Than predict ratings for all pairs (u, i) that are NOT in the training set.
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    # Evaluate Top-N Reccomendations
    top_n = get_top_n(predictions, n=N)
    hr, arhr = hit_ratio(top_n, N, test_data)
    return rmse, mae, hr, arhr


def enhance_trainset(city, review_th, trainset, group_list):
    _, _, df_reviews = dp.get_city_data(city, review_th)
    df_reviews = df_reviews[["user_id", "business_id", "stars"]]
    df = df_reviews[["user_id"]].drop_duplicates()
    df["group_id"] = df.index + len(group_list) + 1
    df_reviews = df_reviews.merge(df, on="user_id", how="inner")
    df_reviews = df_reviews.rename(columns={"stars": "rating"})
    df_reviews = df_reviews[["group_id", "business_id", "rating"]]
    trainset = pd.concat([trainset, df_reviews], sort=True)
    return trainset
