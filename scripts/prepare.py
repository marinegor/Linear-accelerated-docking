from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd
import pickle
import multiprocessing 
from dask.delayed import delayed
import functools
import pandas as pd
import numpy as np


class RawData:
    raw_data = '/storage/marinegor/github/wiselydock-server/data'
    AmpC = f'{raw_data}/AmpC/AmpC_screen_table.csv.gz'
    D4 = f'{raw_data}/D4/D4_screen_table.csv.gz'
    D4_small = f'{raw_data}/D4_1M/D4_screen_table__unique_dropna_213_nonempty_filtered_1M.csv.source'


def report(func):
    functools.wraps(func)

    def inner(*a, **kwa):
        print(f'Executing {func.__name__}')
        return func(*a, **kwa)
    return inner


@report
def read_raw_data(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)

@delayed
def fps_for_arr(arr: list[str]):
    func = lambda s: rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprintAsNumPy(Chem.MolFromSmiles(s))
    return [func(s) for s in arr]
    

def generate_fingerprints(list_of_smiles: list[str], n_parts: int = 32):
    computations = delayed([fps_for_arr(arr) for arr in np.array_split(list_of_smiles, n_parts)])
    fps = computations.compute(scheduler='processes', n_workers=n_parts)
    fps = [elem for subarr in fps for elem in subarr]
    return fps


class Database:
    def __init__(self, which: str, **kwargs):
        if which.lower() == 'd4':
            path = RawData.D4
        elif which.lower() == 'ampc':
            path = RawData.AmpC
        elif which.lower() == 'd4_small':
            path = RawData.D4_small
        else:
            raise ValueError('wrong dataset name')
        df = read_raw_data(path, **kwargs).dropna(subset='dockscore')
        self.df = df
        self.prefix = f'{which.lower()}'

    @report
    def generate_fingerprints(self):
        self.df['fps'] = generate_fingerprints(self.df.smiles.values)

    @report
    def to_pickle(self, path: str = None) -> str:
        fout = f'{RawData.raw_data}/{self.prefix}.pickle'
        with open(fout, 'wb') as f:
            pickle.dump(self, f)
        return fout

    @classmethod
    def from_pickle(cls, which: str) -> 'Database':
        if which.lower() == 'd4':
            path = RawData.D4
        elif which.lower() == 'ampc':
            path = RawData.AmpC
        elif which.lower() == 'd4_small':
            path = RawData.D4_small
        else:
            raise ValueError('wrong dataset name')
        prefix = f'{which.lower()}'
        path = f'{RawData.raw_data}/{prefix}.pickle'
        with open(path, 'rb') as fin:
            return pickle.load(fin)

if __name__ == '__main__':
    for name in ['D4', 'AmpC']:
        db = Database(name, engine='pyarrow')
        db.generate_fingerprints()
        db.df.to_parquet(f'{RawData.raw_data}/{db.prefix}.parquet')
