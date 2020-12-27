import pandas as pd
from collections import Counter


def generate_csv():
    print("Converting json to csv...")
    for f in ["business", "user", "tip"]:
        data = pd.read_json(f + ".json", lines=True)
        data.to_csv(f + ".csv", header=True, index=False)
        print("Successfully converted for:", f)

    # Review file needs to be processed in chunks
    df_reviews = pd.DataFrame()
    good_columns = ['user_id', 'business_id', 'stars', 'date']
    data = pd.read_json("review.json", lines=True, chunksize=100000)
    for chunk in data:
        chunk = chunk[good_columns]
        df_reviews = df_reviews.append(chunk, ignore_index=True)
    df_reviews.to_csv("reviews.csv", header=True, index=False)


def get_restaurant_data():
    print("Extracting data for resturants....")
    df_business = pd.read_csv("business.csv")
    # Drop rows without categories
    df_business = df_business.dropna(subset=['categories'])
    df_business = df_business.reset_index(drop=True)
    # Read category list
    df_list = pd.read_csv("yelp_category_list.csv", header=None)
    cat_list = df_list[0].tolist()
    df_business["categories"] = df_business["categories"].apply(lambda x: x.split(" "))
    # Compare tags to extract restaurants
    df_business = df_business[df_business["categories"].apply(lambda x: len(set(x) & set(cat_list)) > 0)]
    df_business.to_csv("restaurant_business.csv", header=True, index=False)
    print("Restaurants extracted successfully...")

    df_reviews = pd.read_csv("reviews.csv")
    df_reviews = df_reviews.merge(df_business[["business_id"]], on="business_id", how="inner")
    df_reviews.to_csv("restaurant_reviews.csv", header=True, index=False)
    print("Reviews extracted successfully...")

    df_user = pd.read_csv("user.csv", usecols=['average_stars', 'fans', 'friends', 'name', 'review_count', 'user_id', 'yelping_since'])
    user_list = list(set(df_reviews["user_id"].tolist()))
    df_user = df_user[df_user["user_id"].isin(user_list)]
    df_user.to_csv("restaurant_users.csv", header=True, index=False)
    print("Users extracted successfully...")


def get_city_data(city, review_th):
    df_business = pd.read_csv("restaurant_business.csv")
    df_business = df_business[df_business["city"].isin(city)]
    df_reviews = pd.read_csv("restaurant_reviews.csv")
    df_reviews = df_reviews.merge(df_business[["business_id"]], on="business_id", how="inner")
    user_list = Counter(df_reviews["user_id"])
    user_list = list(user_list.items())
    user_list = [user for user, count in user_list if count >= review_th]
    df_reviews = df_reviews[df_reviews["user_id"].isin(user_list)]
    business_list = list(set(df_reviews["business_id"].tolist()))
    df_business = df_business[df_business["business_id"].isin(business_list)]
    df_user = pd.read_csv("restaurant_users.csv")
    df_user = df_user[df_user["user_id"].isin(user_list)]
    df_user["friends"] = df_user["friends"].apply(lambda x: x.split(", "))
    print("Data for the city has been retrieved...")
    df_reviews = df_reviews.reset_index(drop=True)
    df_user = df_user.reset_index(drop=True)
    df_business = df_business.reset_index(drop=True)
    return df_user, df_business, df_reviews
