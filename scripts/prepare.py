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
from typing import Callable
from sklearn.model_selection import KFold


def report(func):
    functools.wraps(func)

    def inner(*a, **kwa):
        print(f"Executing {func.__name__}")
        return func(*a, **kwa)

    return inner


class RawData:
    raw_data = "/storage/marinegor/github/wiselydock-server/data"
    AmpC = f"{raw_data}/AmpC/batch"
    D4 = f"{raw_data}/D4/batch"
    D4_small = f"{raw_data}/D4_1M/D4_screen_table__unique_dropna_213_nonempty_filtered_1M.csv.source"


@report
def read_raw_data(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


@delayed
def fps_for_arr(arr: list[str]):
    func = lambda s: rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048
    ).GetFingerprintAsNumPy(Chem.MolFromSmiles(s))
    return [func(s) for s in arr]


def generate_fingerprints(list_of_smiles: list[str], n_parts: int = 32):
    computations = delayed(
        [fps_for_arr(arr) for arr in np.array_split(list_of_smiles, n_parts)]
    )
    fps = computations.compute(scheduler="processes", n_workers=n_parts)
    fps = [elem for subarr in fps for elem in subarr]
    return fps


@delayed
def fingerprints_from_file(filename: str, ids: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(filename).set_index("zincid")
    df = df[df.index.isin(ids)]
    return df


def read_fingerprints_for(
    filenames: list[str], list_of_ids: list[str], n_parts: int = 16
):
    computations = delayed(
        [fingerprints_from_file(filename, list_of_ids) for filename in filenames]
    )
    dfs = computations.compute(scheduler="processes", n_workers=n_parts)
    return pd.concat(dfs)


@delayed
def apply_model_to(filename: str, model: "OneShotModel") -> pd.DataFrame:
    df = pd.read_parquet(filename).set_index("zincid")
    X = np.vstack(df.fps.values)
    y = model.predict(X)
    df["dockscore_pred"] = y
    df = df[["dockscore_pred"]]
    return df


def apply_single_model_to(filenames: list[str], model, n_parts: int = 12) -> pd.DataFrame:
    computations = delayed([apply_model_to(filename, model) for filename in filenames])
    dfs = computations.compute(scheduler="processes", n_workers=n_parts)
    return pd.concat(dfs)


class Database:
    def __init__(self, which: str, fingerprints: bool = True):
        if which.lower() == "d4":
            path = RawData.D4
        elif which.lower() == "ampc":
            path = RawData.AmpC
        elif which.lower() == "d4_small":
            path = RawData.D4_small
        else:
            raise ValueError("wrong dataset name")

        self.path = Path(path).resolve()
        self.files = [
            f
            for f in self.path.iterdir()
            if f.is_file() and f.stem.startswith("batch") and f.suffix == ".csv"
        ]
        self.prefix = f"{which.lower()}"
        if fingerprints:
            self.generate_fingerprints()

    def read_df(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(
            path, engine="pyarrow", names=["zincid", "smiles", "dockscore"]
        ).dropna(subset="dockscore")

    @report
    def generate_fingerprints(self, force: bool = False):
        parquet_files = []
        for f in tqdm(self.files, desc="generating fingerprints"):
            fout = f"{f}.parquet"
            if force or not Path(fout).exists():
                df = self.read_df(f)
                df["fps"] = generate_fingerprints(df.smiles.values)
                df.to_parquet(fout)
            parquet_files.append(fout)
        self.parquet_files = parquet_files

    @report
    def get_dataframe_for(self, ids: list[str]) -> pd.DataFrame:
        if not hasattr(self, "parquet_files"):
            self.generate_fingerprints()
        return read_fingerprints_for(self.parquet_files, ids)

    def index(self) -> np.ndarray:
        get_index = delayed(lambda filename: pd.read_parquet(filename).zincid.values)
        computations = delayed([get_index(f) for f in self.parquet_files]).compute(n_workers=16)
        return np.hstack(computations)

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

        self.models: dict[int, object] = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        kf = KFold(n_splits=self.n_splits)
        models = delayed(
            [
                fit_model_to_data(self._regressor_factory(), X[xidx_t], y[yidx_t])
                for (xidx_t, _), (yidx_t, _) in kf.split(X, y)
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
        self.iteration = 0

    def add_iteration(self, df: pd.DataFrame):
        assert hasattr(df, "zincid")
        assert hasattr(df, "dockscore")
        assert hasattr(df, "fps")

        X = np.vstack(df.fps.values)
        y = df.dockscore.values

        model = self._oneshot_factory()
        model.fit(X, y)

        self.models[self.iteration] = model
        self.iteration += 1

    def apply_to(self, db: Database) -> pd.DataFrame:
        if self.regime == 'LastModel':
            return self._apply_with_lastmodel(db)
        else:
            raise NotImplementedError

    def _apply_with_lastmodel(self, db: Database) -> pd.DataFrame:
        model = self.models[self.iteration - 1]
        return apply_single_model_to(db.parquet_files, model)
    

db = Database('AmpC')
model = ActiveLearningModel()

# if __name__ == '__main__':
#    for name in ['AmpC', 'D4']:
    #    db = Database(name)
    #    db.generate_fingerprints()
