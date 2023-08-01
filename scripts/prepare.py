from utils import Database, InMemoryDatabase

if __name__ == "__main__":
    # for name in ("D4_medium",):
    # for name in ("D4_small", "D4", "AmpC"):
    for name in ("D4", "AmpC"):
        print(f"Preparing {name}")
        # db = InMemoryDatabase(name, chunksize=1_000_000, dropna=True)
        db = Database(name, chunksize=10_000_000, dropna=True, initialize=True)
