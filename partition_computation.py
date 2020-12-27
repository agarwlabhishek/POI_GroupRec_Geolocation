import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
from haversine import haversine
import matplotlib.pyplot as plt
import data_preprocessing as dp
import re
pd.options.mode.chained_assignment = None


def get_length(x):
    t = MultiPoint(x).convex_hull.bounds
    t = haversine((t[0], t[1]), (t[2], t[3]))
    return t


def cluster_coordinates(coords, eps, max_length, max_iter):
    n_iter = 0
    old_eps = eps
    new_eps = eps
    while (n_iter < max_iter):
        n_iter += 1
        epsilon = new_eps / 6371.0088
        cluster = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
        cluster_labels = cluster.labels_

        df_cluster = pd.DataFrame(coords, columns=["lat", "lng"])
        df_cluster["cluster_id"] = cluster_labels
        df_cluster["coords"] = df_cluster[["lat", "lng"]].apply(lambda x: (x[0], x[1]), axis=1)
        df_cluster = df_cluster.groupby("cluster_id")["coords"].apply(list).to_frame().reset_index()
        df_cluster["length"] = df_cluster["coords"].apply(lambda x: get_length(x))
        # df_cluster["count"] = df_cluster["coords"].apply(lambda x: len(x))
        if np.max(df_cluster["length"].tolist()) > max_length:
            if n_iter == 1:
                new_eps = new_eps/3
            else:
                break
        else:
            old_eps = new_eps
            new_eps = old_eps * 1.1

    epsilon = old_eps / 6371.0088
    cluster = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    cluster_labels = cluster.labels_
    # print('Number of clusters:', len(set(cluster_labels)))
    return cluster_labels


def compute_topK_partitions(city, review_th, group_list, group_train, group_test, K, eps, max_length, max_iter):
    # Get restaurants and reviews for given city
    _, df_business, df_reviews = dp.get_city_data(city, review_th)

    # Get all the individual reviews in train and test data
    test = [(member, item) for group, item in group_test for member in group]
    test = pd.DataFrame(test, columns=["user_id", "business_id"])
    test_reviews = df_reviews.merge(test, on=["user_id", "business_id"], how="inner")
    test_reviews = test_reviews.drop_duplicates()
    train_reviews = df_reviews.merge(test_reviews, indicator=True, how="outer")
    train_reviews = train_reviews[train_reviews["_merge"] == "left_only"].drop("_merge", axis=1)

    print("Computing Top-K Individual Partitions...")
    df_partition = pd.DataFrame()
    user_list = list(set([member for group, sim in group_list for member in group]))
    for user_id in user_list:
        user_reviews = df_reviews[df_reviews["user_id"] == user_id]
        # Build clusters based on individual location history
        user_history = user_reviews.groupby("business_id").aggregate({"stars": ["count", "mean"]})
        user_history.columns = user_history.columns.droplevel(0)
        user_history = user_history.merge(df_business[["business_id", "latitude", "longitude"]], on="business_id", how="inner")
        user_history["cluster_id"] = cluster_coordinates(
                user_history[["latitude", "longitude"]].values, eps, max_length, max_iter)

        partition = user_history.groupby("cluster_id")["business_id"].apply(list).to_frame().reset_index()
        user_history["coords"] = user_history[["latitude", "longitude"]].apply(lambda x: (x[0], x[1]), axis=1)
        temp = user_history.groupby("cluster_id")["coords"].apply(list).to_frame().reset_index()
        partition = partition.merge(temp, on="cluster_id", how="inner")

        temp = user_history.groupby("cluster_id")["count"].sum().to_frame().reset_index()
        partition = partition.merge(temp, on="cluster_id", how="inner")
        temp = user_history.groupby("cluster_id")["mean"].mean().to_frame().reset_index()
        partition = partition.merge(temp, on="cluster_id", how="inner")
        partition["cluster_size"] = partition["business_id"].apply(lambda x: len(x))
        # Choose Top-K partitions
        partition = partition.sort_values(by=["cluster_size", "count", "mean"], ascending=False)
        partition = partition[:K]

        # Store information for each Top-K partition
        partition["length"] = partition["coords"].apply(lambda x: get_length(x))
        partition["centroid"] = partition["coords"].apply(lambda x: np.array(MultiPoint(x).centroid))
        partition["user_id"] = user_id
        partition = partition.drop("cluster_id", axis=1)
        df_partition = pd.concat([df_partition, partition], sort=True)

    df_partition["length"] = df_partition["length"].apply(lambda x: np.round(x, 2))
    temp = df_partition.groupby("user_id")["length"].mean().tolist()
    print("Mean Length:", np.mean(temp))
    print("Std Length:", np.std(temp))

    plt.figure()
    plt.hist(temp, density=False, bins=int(round((np.max(temp)-np.min(temp))*10)), edgecolor='black', linewidth=0.5)
    plt.ylabel('Number of Users')
    plt.xlabel('Average Length Per Cluster (KMS)')
    plt.savefig("partition_length_" + str(K) + ".pdf", bbox_inches='tight', dpi=900)

    return df_partition


def cluster_partitions(group, eps, max_iter):
    old_eps = eps
    new_eps = eps
    n_iter = 0
    while (n_iter < max_iter):
        n_iter += 1
        epsilon = new_eps / 6371.0088
        cluster = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(group[["lat", "lng"]].values))
        cluster_labels = cluster.labels_
        group["cluster_id"] = cluster_labels
        cluster_check = group.groupby("cluster_id")["user_id"].apply(list).to_frame().reset_index()
        cluster_check["check"] = cluster_check["user_id"].apply(lambda x: 1 if len(set(x)) < len(x) and len(set(x)) != 1 else 0)
        if (np.sum(cluster_check["check"].tolist())) != 0:
            if n_iter == 1:
                new_eps = new_eps*0.9
            else:
                break
        else:
            old_eps = new_eps
            new_eps = old_eps*1.05

    epsilon = old_eps / 6371.0088
    cluster = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(group[["lat", "lng"]].values))
    cluster_labels = cluster.labels_

    return cluster_labels


def compute_group_partition(group_list, df_partition, K):
    not_found = []
    df_result = pd.DataFrame()
    print("Computing Top-M Group Partitions...")
    for grp, sim in group_list:
        df_group = df_partition[df_partition["user_id"].isin(grp)]

        df_group["lat"] = df_group["centroid"].apply(lambda x: x[0])
        df_group["lng"] = df_group["centroid"].apply(lambda x: x[1])

        df_group["cluster_id"] = cluster_partitions(df_group[["user_id", "lat", "lng"]], 1, 50)

        partition = df_group.groupby("cluster_id")["user_id"].apply(list).to_frame().reset_index()
        partition["user_count"] = partition["user_id"].apply(lambda x: len(list(set(x))))

        temp = df_group.groupby("cluster_id")["mean"].mean().to_frame().reset_index()
        partition = partition.merge(temp, on="cluster_id", how="inner")

        df_group["business_count"] = df_group["business_id"].apply(lambda x: len(x))
        temp = df_group.groupby("cluster_id")["business_count"].sum().to_frame().reset_index()
        partition = partition.merge(temp, on="cluster_id", how="inner")

        temp = df_group.groupby("cluster_id")["coords"].apply(list).to_frame().reset_index()
        partition = partition.merge(temp, on="cluster_id", how="inner")

        temp = df_group.groupby("cluster_id")["length"].max().to_frame().reset_index()
        partition = partition.merge(temp, on="cluster_id", how="inner")

        partition = partition.sort_values(by=['user_count', 'business_count', 'mean'], ascending=False)
        partition = partition[partition["user_count"] > 1]
        if len(partition) > 0:
            partition = partition[:K]

            partition["coords"] = partition["coords"].apply(lambda x: [subitem for item in x for subitem in item])
            partition["hull_length"] = partition["coords"].apply(lambda x: get_length(x))
            partition["partition_centroid"] = partition["coords"].apply(lambda x: np.array(MultiPoint(x).convex_hull.centroid))
            partition["partition_radius"] = partition[["hull_length", "length"]].apply(lambda x: (x[0] + x[1])/2, axis=1)

            grp = tuple(grp)
            partition["group"] = len(partition)*[grp]
            partition["similarity"] = sim
            df_result = pd.concat([df_result, partition], sort=True)
        else:
            not_found.append((grp, sim))

    temp = df_result["partition_radius"].tolist()
    print("Mean Length:", np.mean(temp))
    print("Std Length:", np.std(temp))

    temp = np.round(temp, 2)
    plt.figure()
    plt.hist(temp, density=False, bins=int(round((np.max(temp)-np.min(temp))*5)), edgecolor='black', linewidth=0.5)
    plt.ylabel('Number of Clusters')
    plt.xlabel('Average Radius Per Cluster (KM)')
    plt.savefig("common_partition_length_" + str(K) + ".pdf", bbox_inches='tight', dpi=900)

    df_result["group_count"] = df_result["group"].apply(lambda x: len(x))
    df_result["user_pt"] = df_result[["group", "user_id"]].apply(lambda x: len(set(x[1]))/len(set(x[0])), axis=1)
    temp = df_result.groupby(["group_count", "user_pt"])["partition_radius"].count().to_frame().reset_index()
    temp = temp.merge(df_result.groupby(["group_count"])["length"].count().to_frame().reset_index(), on="group_count", how="inner")
    temp["percent"] = temp["partition_radius"]/temp["length"]
    print(temp[['group_count', 'user_pt', 'percent']])

    df_result = df_result[["group", "similarity", "partition_centroid", "partition_radius", "user_id", "user_count"]]
    return df_result, not_found


def check_distance(x, y, r):
    if haversine(x, y) <= r:
        return 1
    else:
        return 0


def check_partition_coords(partition_list, df_restaurant):
    result = []
    for coords, radius in partition_list:
        df_res = df_restaurant.copy(deep=True)
        df_res["check"] = df_res[["latitude", "longitude"]].apply(lambda x: check_distance((x[0], x[1]), (coords[0], coords[1]), radius), axis=1)
        df_res = df_res[df_res["check"] == 1]
        if len(df_res) > 0:
            result += df_res["business_id"].to_list()
    return result


def get_partition_restaurants(city, review_th, common_partition):
    # Get reviews for the given city
    _, df_business, _ = dp.get_city_data(city, review_th)
    df_business = df_business[["business_id", "latitude", "longitude"]]
    common_partition["partition_centroid"] = common_partition["partition_centroid"].apply(lambda x: [float(i) for i in re.split('[],;[ ]', x) if len(i) > 0])
    common_partition["partition_info"] = common_partition[["partition_centroid", "partition_radius"]].apply(lambda x: (x[0], x[1]), axis=1)
    df_group = common_partition.groupby("group")["partition_info"].apply(list).to_frame().reset_index()
    df_group["valid_business"] = df_group["partition_info"].apply(lambda x: check_partition_coords(x, df_business))
    df_group["group_id"] = df_group.index
    return df_group
