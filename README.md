# pytorch-projection-qm
A memory-optimised subset of tf-projection-qm implemented in PyTorch. See https://github.com/amreis/tf-projection-qm for a superset of these metrics.

The metrics included are:
        "trustworthiness": trustworthiness.item(),
        "continuity": continuity.item(),
        "scale_normalized_stress": scale_normalized_stress.item(),
        "neighborhood_hit": neighborhood_hit.item(),
        "true_neighbors": true_neighbors.item(),
        "distance_consistency": distance_consistency.item(),
        "procrustes_error": procrustes_error.item(),
        "pearson_correlation": pearson_correlation.item(),
