from utils import Database

if __name__ == "__main__":
    from dask.distributed import Client
    client = Client(n_workers=48, threads_per_worker=2, heartbeat_interval=1_000_000_000, timeout=10_000_000)
    # for name in ("D4", "AmpC"):
    for name in ("D4_small",):
        print(f"Preparing {name}")
        db = Database(name, chunksize=50_000, initialize=True, dropna=True)
