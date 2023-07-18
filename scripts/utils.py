from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd
from dask.delayed import delayed
import functools
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearnex.linear_model import LinearRegression
from typing import Callable, Iterable
from sklearn.model_selection import KFold

from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap


class RawDataPath:
    raw_data = "/storage/marinegor/github/wiselydock-server/data"
    AmpC = f"{raw_data}/AmpC/"
    D4 = f"{raw_data}/D4"
    D4_small = f"{raw_data}/D4_small"


class Database:
    def __init__(self, which: str, chunksize: int = 50_000, read_raw_data: bool = True):
        if which.lower() == "d4":
            path = RawDataPath.D4
        elif which.lower() == "ampc":
            path = RawDataPath.AmpC
        elif which.lower() == "d4_small":
            path = RawDataPath.D4_small
        else:
            raise ValueError("wrong dataset name")

        self.path = Path(path).resolve()
        self.which = which
        self.raw = f"{self.path}/raw.csv"
        self.chunksize = chunksize

        if read_raw_data:
            self.read_raw_data()
        self.write_columns_in_batches()
        self.add_fingerprints()

    @timing
    def read_raw_data(self):
        df = pd.read_csv(self.raw, engine="pyarrow")
        self.df_raw = df
        return df

    @timing
    def _write_to_disk(
        self, out_filename: str, df: pd.DataFrame, column: str, force: bool = False
    ):
        if force or not Path(out_filename).exists():
            df[[column]].to_parquet(out_filename)

    @timing
    def write_columns_in_batches(self):
        chunksize = self.chunksize
        df = self.read_raw_data()
        columns = df.columns.values

        for column in columns:
            Path(f"{self.path}/{column}").mkdir(exist_ok=True)

        n_splits = int(len(df) / chunksize)
        for column in columns:
            computations = delayed(
                [
                    self._write_to_disk(
                        f"{self.path}/{column}/batch_{idx:08}.parquet", chunk, column
                    )
                    for idx, chunk in enumerate(np.array_split(df, n_splits))
                ]
            )
            computations.compute(n_workers=32)

    @timing
    def get_filenames_for(self, column: str) -> list[Path]:
        return sorted(Path(f"{self.path}/{column}").iterdir())

    @timing
    def read_column(self, column: str, idx: np.ndarray = None) -> np.ndarray:
        if column == "fingerprints":
            func = lambda filename: _read_fingerprints(filename=filename, idx=idx)
            agg = np.vstack
        else:
            func = lambda filename: _read_column_with_idx(
                filename=filename, column=column, idx=idx
            )
            agg = np.hstack
        func = delayed(func)
        batches = self.get_filenames_for(column)
        tasks = delayed([func(batch) for batch in batches])
        results = tasks.compute(n_workers=32)
        return agg(results)

    @timing
    def add_fingerprints(self, column="smiles"):
        out_folder = self.path / "fingerprints"
        out_folder.mkdir(exist_ok=True)

        tasks = delayed(
            [
                _fingerprint_helper(
                    input_filename=batch,
                    output_filename=_replace_column_name(batch, "fingerprints"),
                )
                for batch in self.get_filenames_for(column)
            ]
        )
        tasks.compute(n_workers=32)

    def _execute(
        self, df_func: Callable, agg_func: Callable, n_parts: int = 12, columns=None
    ):
        df_func = delayed(df_func)
        results = delayed(
            [
                df_func(self._read_df(filename, columns=columns))
                for filename in self.parquet_files
            ]
        ).compute(n_workers=n_parts)
        return agg_func(results)

    def get_random_batch(self, batchsize: int) -> tuple[np.ndarray, np.ndarray]:
        scores = self.read_column("dockscore")
        idx = np.arange(len(scores))
        rv_idx = np.random.choice(idx, size=(batchsize,))
        rv_scores = scores[rv_idx]
        return rv_idx, rv_scores

    def read_dataframe_for(
        self,
        ids: list[int],
        columns: list[str] = ["zincid", "dockscore", "fingerprints"],
    ) -> pd.DataFrame:
        rv = pd.DataFrame()
        for column in columns:  # FIXME
            batches = self.get_filenames_for(column)
            tasks = delayed(
                [_read_subset_helper(batch, ids, column) for batch in batches]
            )
            results = tasks.compute(n_workers=32)
            rv[column] = np.hstack(results)
        return rv


def _replace_column_name(p: Path, target: str) -> Path:
    return p.parent.parent / target / p.name


@delayed
def _read_subset_helper(filename: str, ids: list[int], column: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    return df[df.index.isin(ids)][column].values


@delayed
def _fingerprint_helper(
    input_filename: str,
    output_filename: str,
    column: str = "smiles",
    force: bool = False,
):
    if force or not Path(output_filename).exists():
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        smiles2fp = lambda s: generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s))
        df = pd.read_parquet(input_filename)
        arr = np.vstack([smiles2fp(s) for s in df[column]])
        arr_as_df = pd.DataFrame(
            arr, columns=[f"fp_{idx:04}" for idx in range(arr.shape[1])]
        )
        arr_as_df.to_parquet(output_filename)


def _read_column_with_idx(filename: str, column: str, idx: np.ndarray) -> np.ndarray:
    df = pd.read_parquet(filename)
    if idx is not None:
        df = df[df.index.isin(idx)]
    return df[column].values


def _read_fingerprints(filename: str, idx: np.ndarray) -> np.ndarray:
    df = pd.read_parquet(filename)
    if idx is not None:
        df = df[df.index.isin(idx)]
    return df.values


@delayed
def apply_model_to(filename: str, model: "OneShotModel") -> pd.DataFrame:
    df = pd.read_parquet(filename).set_index("zincid")
    X = np.vstack(df.fps.values)
    y = model.predict(X)
    df["dockscore_pred"] = y
    df = df[["dockscore_pred"]]
    return df


def apply_single_model_to(
    filenames: list[str], model, n_parts: int = 12
) -> pd.DataFrame:
    computations = delayed([apply_model_to(filename, model) for filename in filenames])
    dfs = computations.compute(scheduler="processes", n_workers=n_parts)
    return pd.concat(dfs)


@delayed
def fit_model_to_data(model, X, y):
    model.fit(X, y)
    return model


class OneShotModel:
    def __init__(self, n_splits: int = 5, regressor_factory: Callable = None):
        self.n_splits = n_splits

        if regressor_factory is None:
            regressor_factory = lambda: LinearRegression(n_jobs=8)
        self._regressor_factory = regressor_factory

    def fit(self, X: np.ndarray, y: np.ndarray):
        kf = KFold(n_splits=self.n_splits)
        models = delayed(
            [
                fit_model_to_data(self._regressor_factory(), X[idx], y[idx])
                for idx, _ in kf.split(X, y)
            ]
        )
        models = models.compute(n_workers=self.n_splits)
        self.models = models
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        ypreds = np.zeros(X.shape[0])
        for model in self.models:
            ypreds += model.predict(X)
        ypreds /= len(self.models)
        return ypreds


class ActiveLearningModel:
    def __init__(
        self,
        regime: str = "LastModel",
        oneshot_factory: Callable = None,
    ):
        if regime not in ("LastModel", "MeanRank", "TopFromEveryModel"):
            raise ValueError(f"Wrong {regime=}")
        self.regime = regime

        if oneshot_factory is None:
            oneshot_factory = lambda: OneShotModel()
        self._oneshot_factory = oneshot_factory

        self.models: dict[int, object] = {}
        self.seen_molecules: dict[int, np.ndarray] = {}
        self.iteration = 0

    def add_iteration(
        self, scores: np.ndarray, ids: np.ndarray, fingerprints: np.ndarray
    ):
        X = fps
        y = scores

        model = self._oneshot_factory()
        model.fit(X, y)

        self.models[self.iteration] = model
        self.seen_molecules[self.iteration] = [ids]
        self.iteration += 1

    def apply_to(self, db: Database) -> pd.DataFrame:
        if self.regime == "LastModel":
            return self._apply_with_lastmodel(db)
        else:
            raise NotImplementedError

    def _apply_with_lastmodel(self, db: Database) -> pd.DataFrame:
        model = self.models[self.iteration - 1]
        return apply_single_model_to(db.parquet_files, model)


if __name__ == "__main__":
    batchsize = 10_000
    num_iterations = 10
    db = Database("D4_small")
    model = ActiveLearningModel()

    idx, scores = first_batch = db.get_random_batch(batchsize=batchsize)
    fps = db.read_column("fingerprints")[idx]

    # for i in range(num_iterations):
    #     model.add_iteration()
