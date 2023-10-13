from utils import ActiveLearningModel, InMemoryDatabase, Database, IterativeDatabase
import numpy as np
import pickle

if __name__ == "__main__":
    for name in ("D4", "AmpC"):
        batchsize = 20_000
        num_iterations = 100
        model = ActiveLearningModel(regime="MeanRank")
        db = IterativeDatabase(name)

        idx, scores = first_batch = db.get_random_batch(batchsize=batchsize)
        fps = db.read_column("fingerprints", idx=idx)
        for i in range(num_iterations):
            np.save(f"{name}_{i}", idx)

            model.add_iteration(scores, idx=idx, fingerprints=fps)

            predicted_scores = model.get_preds_for(db)

            idx = model.select_top_k(db, predicted_scores, top_k=batchsize)
            fps = db.read_column("fingerprints", idx=idx)
            scores = db.read_column("dockscore", idx=idx)

            for model_idx, submodel in enumerate(model.models):
                with open(f"{name}_{i}_model_{model_idx}.pickle", "wb") as fout:
                    pickle.dump(submodel, fout)
