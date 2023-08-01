from utils import ActiveLearningModel, InMemoryDatabase, Database
import numpy as np

if __name__ == "__main__":
    for name in ("D4",):
    # for name in ("D4_medium",):
        batchsize = 10_000
        num_iterations = 100
        model = ActiveLearningModel(regime="MeanRank")
        db = Database(name, chunksize=10_000)

        idx, scores = first_batch = db.get_random_batch(batchsize=batchsize)
        fps = db.read_column("fingerprints", idx=idx)

        for i in range(num_iterations):
            np.save(f"{name}_{i}", idx)
            
            model.add_iteration(scores, idx=idx, fingerprints=fps)

            predicted_scores = model.get_preds_for(db)

            idx = model.select_top_k(db, predicted_scores, top_k=batchsize)
            fps = db.read_column("fingerprints", idx=idx)
            scores = db.read_column("dockscore", idx=idx)

        break
