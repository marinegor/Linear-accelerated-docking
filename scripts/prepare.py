from utils import Database, InMemoryDatabase, IterativeDatabase

if __name__ == "__main__":
    # for name in ("D4_medium",):
    # for name in ("D4_small", ):
    for name in ("D4", "AmpC"):
        print(f"Preparing {name}")
        db = IterativeDatabase(name, chunksize=10_000_000, dropna=True, force=True)
