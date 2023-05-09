import sklearn as sk
from itertools import product
from random import sample, seed
import pickle
import json
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from sklearn.base import BaseEstimator, RegressorMixin
from glob import glob
import numpy as np
import pandas as pd
import os
import argparse
import sys
import datetime as dt

possible_methods = ["MeanRank", "TopFromEveryModel", "LastModel"]
possible_fingerprints = ["Morgan", "AtomPairs"]
possible_models = [
    "LinearRegression",
    "DockingAsPredictor",
    "RandomGaussianRegressor",
    "LinearSVR",
    "LassoCV",
    "RidgeCV",
]
possible_sizes = [2048, 4096]


def loggg(with_dataframe=True):
    def decorator(f):
        if with_dataframe == True:

            def wrapper(dataf, *args, **kwargs):
                tic = dt.datetime.now()
                result = f(dataf, *args, **kwargs)
                toc = dt.datetime.now()
                if hasattr(dataf, "shape") and hasattr(result, "shape"):
                    share_before = dataf.shape
                    shape_after = result.shape
                    added_columns = set(result.columns) - set(dataf.columns)
                    print(
                        f"{f.__name__},  shape {dataf.shape}->{result.shape},  took={toc-tic}"
                    )
                else:
                    print(f"{f.__name__} took={toc-tic}")
                return result

        else:

            def wrapper(*args, **kwargs):
                tic = dt.datetime.now()
                result = f(*args, **kwargs)
                toc = dt.datetime.now()

                print(f"{f.__name__} took={toc-tic}")

        return wrapper

    return decorator


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-path",
        type=str,
        help="path to directory with tables of fingerprints and docking scores",
    )
    parser.add_argument("-out", type=str, help="output directory", default=None)
    parser.add_argument(
        "-dpath",
        "--docking_path",
        type=str,
        help="path to docking result in case of DockingAsPredictor is used",
        default=None,
    )
    parser.add_argument(
        "-pos",
        "--positive_scores",
        type=bool,
        help="whether to leave positive scores or to null",
        default=True,
    )
    parser.add_argument(
        "-thr", "--threshold", type=float, help="percentage of hits", default=1.0
    )
    parser.add_argument("-meth", "--method", type=str, help="prediction mehtod")
    parser.add_argument(
        "-m", "--model", type=str, help="sklearn model", default="LinearRegression"
    )
    parser.add_argument("--folds", type=int, help="number of folds", default=1)
    parser.add_argument(
        "-i", "--iterations", type=int, help="number of iterations in iterative process"
    )
    parser.add_argument(
        "-ts", "--trainsize", type=int, help="full train size for all iterations"
    )
    return parser


@loggg(False)
def error_string(s):
    return f"Choose another {s}. Possible {s}s are:"


@loggg(False)
def check_parser_args(namespace):
    assert os.path.exists(namespace.path), "Wrong path"
    assert namespace.method in possible_methods, (
        error_string("prediction method"),
        possible_methods,
    )
    assert namespace.model in possible_models, (error_string("model"), possible_models)
    return 0


parser = create_parser()
params_ns = parser.parse_args()
check_parser_args(params_ns)


# creation of the output directory


if not params_ns.out:
    params_ns.out = params_ns.path.replace(
        "tables_for_prediction", "prediction_results_iterations"
    )

if not os.path.exists("/".join(params_ns.out.split("/")[:-1])):
    os.mkdir("/".join(params_ns.out.split("/")[:-1]))
    os.mkdir(params_ns.out)
elif not os.path.exists(params_ns.out):
    os.mkdir(params_ns.out)

print(f"Writing output to {params_ns.out}")

# downloading the table and preprocessing

# def load_csv_with_dtypes(filename, sep=','):
#    with open(filename) as fin:
#        for first_line in fin:
#            break
#        column_names = first_line.split(sep)
#    dtype = {col.strip():
#             'bool' if col.startswith('fps')
#              else 'string'
#              for col in column_names
#             if col.strip()
#            }
#    dtype['Score'] = 'float'
#
#    df = pd.read_csv(filename, dtype=dtype, index_col=0)
#    print(df.columns())
#    return df


@loggg()
def load_data_from_glob_with_dtypes(glob_expression):

    # size = int(glob_expression.split('/')[-2].split('_')[1].split('=')[1])
    size = 2048

    dtype = {f"fps_{i}": bool for i in range(size)}
    dtype.update(
        {
            "ZincID": str,
            "Smiles": str,
            "Score": float,
        }
    )

    rv = []
    for csv_filename in glob(glob_expression):
        # print(csv_filename)
        df = pd.read_csv(csv_filename, dtype=dtype)
        rv.append(df)
    df = pd.concat(rv, ignore_index=True)
    return df


@loggg()
def convert_score_as_float(dataf):
    dataf["Score"] = dataf.Score.astype("float")
    return dataf


@loggg()
def add_label(table, percentage=params_ns.threshold):
    name = f"Hit(top {percentage}%)"
    pos = round(percentage * table.shape[0] / 100)
    threshold = table.sort_values("Score").Score[pos]

    table[name] = list(map(lambda x: int(x < threshold), table["Score"]))
    return table


@loggg()
def delete_high_scores(table):
    table = table.dropna(subset=["Score"])
    if params_ns.positive_scores == False:
        table["Score"] = np.where(table["Score"] > 0, 0, table["Score"])
    return table


df = (
    load_data_from_glob_with_dtypes(f"{params_ns.path}/*.csv")
    .pipe(delete_high_scores)
    .set_index("ZincID", drop=True)
    .drop(["Smiles"], axis=1)
    .pipe(add_label)
)

print(df.head())


# models for predictions


class RandomGaussianRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, mean: float = -10, std: float = 10):
        self._mean = mean
        self._std = std

        def RandomGaussianSampler(arr):
            return np.random.normal(loc=mean, scale=std, size=arr.shape[0])

        self._model = RandomGaussianSampler

    def fit(self, X, y):
        return self

    def predict(self, X):
        y = self._model(X)
        return y

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)


class DockingAsPredictor(BaseEstimator, RegressorMixin):
    def __init__(self, glob_expression):
        self._glob_expression = glob_expression

    def fit(self, X, y):

        df = load_data_from_glob_with_dtypes(self._glob_expression).dropna(
            subset=["Score"]
        )

        fps_cols = ["Score"] + [col for col in df.columns if col.startswith("fps")]

        self._scores = {}
        for score, *fps in df[fps_cols].itertuples(index=False):
            key = tuple(fps)
            self._scores[key] = score

        return self

    def predict(self, X):
        def score_search_tuple(fps, storage=self._scores):
            return storage.get(tuple(fps), 0)

        return np.array([score_search_tuple(fps) for _, fps in X.iterrows()])


# iteration algorithms
def top_LastModel(predictions_list, position):

    y_pred = predictions_list[0]
    print(len(y_pred))
    y_pred = sorted(list(enumerate(y_pred)), key=lambda idx_score: idx_score[1])
    indexes = [idx for (idx, score) in y_pred[:position]]

    return indexes


def top_TopFromEveryModel(predictions_list, position):

    indexes = []
    for y_pred in reversed(predictions_list):
        temp_y_pred = sorted(
            [(idx, score) for (idx, score) in enumerate(y_pred) if idx not in indexes],
            key=lambda idx_score: idx_score[1],
        )
        indexes += [
            idx for (idx, score) in temp_y_pred[: position // len(predictions_list)]
        ]

    return indexes


def top_MeanRank(predictions_list, position):

    import numpy as np

    indexes = np.zeros(len(predictions_list[0]))
    for y_pred in predictions_list:
        assert len(y_pred) == len(
            predictions_list[0]
        ), f"Different sizes of prediction lists: {len(predictions_list[0])} and {len(y_pred)}"
        temp_y_pred = sorted(
            list(enumerate(y_pred)), key=lambda idx_score: idx_score[1]
        )
        temp_y_pred = sorted(
            [(idx, rank) for (rank, (idx, score)) in enumerate(temp_y_pred)],
            key=lambda idx_rank: idx_rank[0],
        )
        temp_y_pred = [rank for (idx, rank) in temp_y_pred]

        indexes += np.array(temp_y_pred)

    indexes = sorted(list(enumerate(indexes)), key=lambda idx_rank: idx_rank[1])[
        :position
    ]
    indexes = [idx for idx, rank in indexes]

    return indexes


def top_all_models(predictions_list, position, prediction_method):
    if prediction_method == "LastModel":
        return top_LastModel(predictions_list, position)
    elif prediction_method == "TopFromEveryModel":
        return top_TopFromEveryModel(predictions_list, position)
    elif prediction_method == "MeanRank":
        return top_MeanRank(predictions_list, position)


def write_down_ZINCid(X, y, stratification=params_ns.threshold):
    indexes = sorted(list(enumerate(y)), key=lambda x: x[1])[
        : round(stratification * len(y) / 100)
    ]
    indexes = [x[0] for x in indexes]
    ZINCids = X.index.values[indexes]
    return ZINCids.tolist()


def do_iterations_regressors(
    df,
    out,
    model,
    model_key,
    prediction_method,
    iterations,
    full_size,
    add,
    rand_seed,
    stratification=params_ns.threshold,
):

    if add == True:
        add_str = "add"
    else:
        add_str = "noadd"

    print(f"out={out}")
    print(f"model={model}")
    print(f"model_key={model_key}")
    print(f"prediction_method={prediction_method}")
    print(f"iterations={iterations}")
    print(f"full_size={full_size}")
    print(f"add={add}")
    print(f"seed={rand_seed}")

    initial_train_size = full_size // iterations

    hit_cols = [col for col in df.columns if col.startswith("Hit")]
    X_cols = [col for col in df.columns if col.startswith("fps_")]
    X = df[X_cols]
    y = df.Score.values
    stratify_col = hit_cols[0]

    X, X_val, y, y_val = train_test_split(
        X,
        y,
        train_size=0.75,
        random_state=0,
        shuffle=True,
        stratify=df[stratify_col].values,
    )

    # writing down hits from buffer
    ZINC_hits = {}
    ZINC_hits["buffer"] = write_down_ZINCid(X, y)

    assert len(y) > initial_train_size * iterations, f"Too little data!"

    # information about validation set
    position_val = round(len(y_val) * stratification / 100)
    threshold_y_val = sorted(y_val)[position_val]
    indexes_y_val = [i for i in range(len(y_val)) if (y_val[i] < threshold_y_val)]
    score_val = np.median(y_val)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=initial_train_size,
        random_state=rand_seed,
        shuffle=True,
    )

    # hits in train_set
    ZINC_hits["iter_0"] = list(set(ZINC_hits["buffer"]) & set(X_train.index.values))

    models_list = []
    models_predictions_list = []
    models_val_predictions_list = []

    for iteration_idx in range(iterations):

        filename = (
            f"{out}/"
            f"model={model_key}"
            f"_predictionmethod={prediction_method}"
            f"_trainsize={initial_train_size}"
            f"_col={stratify_col}"
            f"_iterations={iterations}"
            f"_{add_str}"
            f"_idx={iteration_idx}"
            f"_seed={rand_seed}.json"
        )

        if os.path.exists(filename):
            toc = dt.datetime.now()
            print(f"{toc}:Prediction already done\n")
            return None
        else:
            print(f"prediction: {filename}")

        # training new model
        tic = dt.datetime.now()
        my_model = model.fit(X=X_train, y=y_train)
        y_pred = my_model.predict(X_test)
        toc = dt.datetime.now() - tic
        toc = [toc.days, toc.seconds, toc.microseconds]
        models_list.append(my_model)
        if prediction_method != "LastModel":
            models_predictions_list.append(y_pred)
        else:
            models_predictions_list = [y_pred]

        # predictions of the full model
        position = round(stratification / 100 * len(y_test))
        # print(f'Position = {position}')
        indexes_y_full_pred = top_all_models(
            models_predictions_list, position, prediction_method
        )

        y_full_pred = [1 if i in indexes_y_full_pred else 0 for i in range(len(y_test))]

        # performance on the validation dataset
        y_val_pred = my_model.predict(X_val)
        if prediction_method != "LastModel":
            models_val_predictions_list.append(y_val_pred)
        else:
            models_val_predictions_list = [y_val_pred]

        indexes_y_val_pred = top_all_models(
            models_val_predictions_list, position_val, prediction_method
        )

        top_score = len(set(indexes_y_val_pred) & set(indexes_y_val)) / len(
            indexes_y_val
        )
        #         print(top_score)

        score_val_pred = np.median([y_val[indexes_y_val_pred]])

        d = {
            "predictions_single_model": y_pred.tolist(),
            "true_labels": y_test.tolist(),
            "predictions_full_model": y_full_pred,
            "time_for_predictions": toc,
            "top_score_val": top_score,
            "true_and_pred_score": (score_val, score_val_pred),
        }

        # train_set augmentation

        if len(indexes_y_full_pred) >= initial_train_size:
            indexes = sample(indexes_y_full_pred, initial_train_size)
        else:
            print(
                f"Initial size ({initial_train_size}) is lagrer than the number of hits({len(indexes_y_full_pred)})"
            )
            non_indexes = [
                i for i in range(X_test.shape[0]) if i not in indexes_y_full_pred
            ]
            indexes = indexes_y_full_pred + sample(
                non_indexes, initial_train_size - len(indexes_y_full_pred)
            )

        non_indexes = [i for i in range(X_test.shape[0]) if i not in indexes]
        X_add_train, X_test = X_test.iloc[indexes], X_test.iloc[non_indexes]
        y_add_train = y_test[indexes]
        y_test = y_test[non_indexes]

        if add == False:
            X_train = X_add_train
            y_train = y_add_train
        else:
            X_train = pd.concat([X_train, X_add_train])
            y_train = np.concatenate((y_train, y_add_train))

        # hits in train_set - from ZINC_hits score can be obtained
        ZINC_hits[f"iter_{iteration_idx + 1}"] = list(
            set(ZINC_hits["buffer"]) & set(X_add_train.index.values)
        )

        # predictions set updating
        temp_models_predictions_list = []
        for y_pred in models_predictions_list:
            temp_models_predictions_list.append(y_pred[non_indexes])
        models_predictions_list = temp_models_predictions_list

        len_predictions_array = [len(y_pred) for y_pred in models_predictions_list]
        assert (
            len(set(len_predictions_array)) == 1
        ), f"Predictions have different lengths"

        with open(filename, "w") as fin:
            json.dump(d, fin)
        # print(filename, top_score)

    #         with open(filename.replace('.json', '.pickle'),'wb') as modelFile:
    #             pickle.dump(my_model, modelFile)

    ZINC_filename = (
        f"{out}/"
        f"model={model_key}"
        f"_predictionmethod={prediction_method}"
        f"_trainsize={initial_train_size}"
        f"_col={stratify_col}"
        f"_iterations={iterations}"
        f"_{add_str}"
        f"_seed={rand_seed}.json"
    )

    with open(ZINC_filename, "w") as ZINC_f:
        json.dump(ZINC_hits, ZINC_f)

    return 0


model_key = params_ns.model

if params_ns.model == "LinearRegression":

    from sklearn.linear_model import LinearRegression

    model = LinearRegression()

elif params_ns.model == "LassoCV":

    from sklearn.linear_model import LassoCV

    model = LassoCV()

elif params_ns.model == "RidgeCV":

    from sklearn.linear_model import RidgeCV

    model = RidgeCV()

elif params_ns.model == "LinearSVR":

    from sklearn.svm import LinearSVR

    model = LinearSVR()

elif params_ns.model == "RandomGaussianRegressor":

    model = RandomGaussianRegressor()

elif params_ns.model == "DockingAsPredictor":

    glob_expression = f"{params_ns.docking_path}/*.csv"
    model = DockingAsPredictor(glob_expression=glob_expression)


for rand_seed, add in product(range(9, 9 + params_ns.folds), (True, False)):

    do_iterations_regressors(
        df,
        params_ns.out,
        model,
        params_ns.model,
        params_ns.method,
        params_ns.iterations,
        params_ns.trainsize,
        add,
        rand_seed,
        stratification=params_ns.threshold,
    )
