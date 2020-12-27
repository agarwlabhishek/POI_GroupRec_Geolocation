import pandas as pd
import data_preprocessing as dp
import group_formation as gf
import partition_computation as pc
import group_recommender as gr

pd.options.mode.chained_assignment = None


# CONVERT JSON FILES TO CSV
# dp.generate_csv()

# GET DATA SEPECIFIC TO RESTAURANTS
# dp.get_restaurant_data()


# SELECT CITY NAMES TO GENERATE SAMPLE DATA
city = ["Phoenix", "Scottsdale", "Tempe", "Mesa", "Chandler",
        "Carefree", "Gilbert", "Glendale", "Peoria", "Surprise",
        "Goodyear", "Tolleson", "Guadalupe", "Litchfield Park"]
# city = ["Phoenix"]


# FORM GROUPS OF GIVEN SIZE
# city, review_th, min_csize, max_csize, c_th, min_gsize, max_gsize
# gf.compute_groups(city, 20, 5, 10, 0.15, 3, 5)

# SAMPLE GROUPS
# city, co-ratings_count, threshold
group_list = gf.get_groups(city, 4, 1)

# GET DATA CORRESPONDING TO A CITY BASED ON REVIEW COUNT
# city, review_th
# df_user, df_business, df_reviews = dp.get_city_data(city, 20)

# SPLIT GROUPS IN TRAIN AND TEST
# city, review_th, group_list, test_size

group_train, group_test = gf.split_train_test(city, 20, group_list, 1)
K = 5
M = 7

# Compute Top-K indivdual partitions for each user
# city, review_th, group_list, group_train, group_test, K, eps, max_length, max_iter
df_partition = pc.compute_topK_partitions(city, 20, group_list, group_train, group_test, K, 1, 2, 50)

# Compute Top-M group partitions for each group
# group_list, df_partition, M
common_partition, not_found = pc.compute_group_partition(group_list, df_partition, M)

common_partition.to_csv("top_m_common_partitions.csv", index=False)
common_partition = pd.read_csv("top_m_common_partitions.csv")

group_partition = pc.get_partition_restaurants(city, 20, common_partition)
group_partition.to_csv("group_partition.csv", index=False)

group_partition = pd.read_csv("group_partition.csv")
"""
df_check = group_partition.merge(test_dataset, on="group_id", how="inner")
df_check["check"] = df_check[["valid_business", "business_id"]].apply(lambda x: 1 if x[1] in x[0] else 0, axis=1)
print(df_check["check"].sum()/len(df_check))
"""
test_part = gr.get_test_data(city, 20, group_partition, group_test)
train_part = gr.build_group_profile(city, 20, group_partition, group_train)
train_bl, test_bl = gr.get_baseline_data(city, 20, group_train, group_test)
"""
N = 10
rmse, mae, hr, arhr = gr.group_recommendation(train_part, test_part, group_partition, "svd", N)
print(hr, arhr)
"""

df_result = pd.DataFrame(columns=["RS", "N", "ALGO", "RMSE", "MAE", "HR", "ARHR"])
pos = 0
for N in [1, 3, 5, 10, 20, 25, 30, 40, 50]:
    print("N:", N)
    for algo in ["svd", "svdpp", "nmf"]:
        rmse, mae, hr, arhr = gr.group_recommendation(train_part, test_part, group_partition, algo, N)
        df_result.loc[pos] = ["PARTITION", N, algo, round(rmse, 4), round(mae, 4), round(hr, 4), round(arhr, 4)]
        pos += 1
        rmse, mae, hr, arhr = gr.baseline_recommendation(train_bl, test_bl, algo, N)
        df_result.loc[pos] = ["BASELINE", N, algo, round(rmse, 4), round(mae, 4), round(hr, 4), round(arhr, 4)]
        pos += 1
        fname = "result_K_" + str(K) + str("_M_") + str(M) + ".csv"
        df_result.to_csv(fname, index=False)
