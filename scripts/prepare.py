from utils import Database

if __name__ == "__main__":
    for name in ("D4", "AmpC"):
        print(f"Preparing {name}")
        db = Database(name, chunksize=10_000, initialize=True)
