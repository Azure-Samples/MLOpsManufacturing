import mlflow
import argparse
import glob

import pandas as pd
import lightgbm as lgbm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def main(args):
    mlflow.autolog()

    num_boost_round = args.num_boost_round
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "boosting": args.boosting,
        "num_iterations": args.num_iterations,
        "num_leaves": args.num_leaves,
        "num_threads": args.num_threads,
        "learning_rate": args.learning_rate,
        "metric": args.metric,
        "seed": args.seed,
        "verbose": args.verbose,
    }

    files = glob.glob(args.input_data + "/*.csv")
    df = pd.DataFrame()
    for f in files:
        csv = pd.read_csv(f)
        df = df.append(csv)
    print(df.head(5))
    X_train, X_test, y_train, y_test, enc = process_data(df)
    model = train_model(params, num_boost_round, X_train, X_test, y_train, y_test)


def process_data(df):
    X = df.drop(["species"], axis=1)
    y = df["species"]

    enc = LabelEncoder()
    y = enc.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, enc


def train_model(params, num_boost_round, X_train, X_test, y_train, y_test):
    train_data = lgbm.Dataset(X_train, label=y_train)
    test_data = lgbm.Dataset(X_test, label=y_test)

    model = lgbm.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[test_data],
        valid_names=["test"],
    )

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str)
    parser.add_argument("--num-boost-round", type=int, default=10)
    parser.add_argument("--boosting", type=str, default="gbdt")
    parser.add_argument("--num-iterations", type=int, default=16)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.9)
    parser.add_argument("--metric", type=str, default="multi_logloss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    print(f"args.input_data is {args.input_data}")

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)