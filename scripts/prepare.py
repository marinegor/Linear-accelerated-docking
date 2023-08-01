from utils import Database, InMemoryDatabase, IterativeDatabase

if __name__ == "__main__":
    for name in ("D4_medium",):
    # for name in ("D4_small", "D4", "AmpC"):
    # for name in ("D4", "AmpC"):
    # for name in ("D4_small", ):
        print(f"Preparing {name}")
        db = IterativeDatabase(name, chunksize=1_000_000, dropna=True, force=True)
