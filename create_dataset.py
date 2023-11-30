import pandas as pd
import os
import argparse


def load_data(path):
    """Load and merge data from csv files.

    Arguments:
        path {str} -- path to data folder
    """
    subjects = pd.read_csv(os.path.join(path, "classhook_subjects_sample.csv"))
    col_filter = ["id", "subtitles_text"]
    resources = pd.read_csv(os.path.join(path, "classhook_resources_sample.csv"))[
        col_filter
    ]
    resources = resources[resources["subtitles_text"].notnull()]
    resource_subjects = pd.read_csv(
        os.path.join(path, "classhook_resource_subjects_sample.csv")
    )
    dataset = resource_subjects.merge(
        subjects[["id", "name"]], left_on="subject_id", right_on="id"
    ).merge(resources, left_on="resource_id", right_on="id")[
        ["id", "name", "subtitles_text"]
    ]
    return dataset


def preprocess(df):
    """Preprocess df: remove dups, lowercase

    Arguments:
        dataset {pd.DataFrame} -- dataset to preprocess
    """
    df = df[df["subtitles_text"].duplicated(keep=False)]
    df = df[df["subtitles_text"].apply(len) >= 100]
    df["subtitles_text"] = df["subtitles_text"].str.lower()

    return df


def select(df, num_subjects):
    """Select subjects with most samples

    Arguments:
        df {pd.DataFrame} -- dataset to select from
        num_subjects {int} -- number of subjects to select
    """
    top = df["name"].value_counts()[:num_subjects]
    df = df[df["name"].isin(top.index)]
    one_hot = pd.get_dummies(df["name"]).astype("int8")
    df = pd.concat([df, one_hot], axis=1)
    df.drop(columns=["name"], inplace=True)
    df = df.groupby("id").sum().reset_index()
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str)
    parser.add_argument("-n", "--num_subjects", type=int)
    parser.add_argument("-o", "--output_path", type=str)
    args = parser.parse_args()
    num_subjects = args.num_subjects
    dataset = load_data(args.input_path)
    df = preprocess(dataset)
    result = select(df, num_subjects)
    result.to_csv(args.output_path, index=False, header=True)
