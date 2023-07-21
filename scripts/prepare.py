from utils import Database

if __name__ == "__main__":
    from dask.distributed import Client
    client = Client(n_workers=48, threads_per_worker=2)
    for name in ("D4", "AmpC"):
        print(f"Preparing {name}")
        db = Database(name, chunksize=50_000, initialize=True, dropna=True)
