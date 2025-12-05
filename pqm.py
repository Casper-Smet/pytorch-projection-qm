# These metrics are implemented following Machado et al.'s tf-projection-qm library.

import torch


@torch.no_grad()
def projection_quality_metrics(
    X: torch.Tensor,
    Y: torch.Tensor,
    k: int,
    labels: torch.LongTensor = None,
    n_classes: int = None,
    eps: float = 1e-12,
):
    n, d_orig = X.shape
    d_emb = int(Y.shape[1])
    k = min(k, n)

    # 1) Pairwise distances
    D_high = torch.cdist(X, X, p=2)  # (n, n)
    D_low = torch.cdist(Y, Y, p=2)  # (n, n)

    # 2) Full sorts & ranks for Trustworthiness/Continuity
    idx_high = torch.argsort(D_high, dim=1)
    idx_low = torch.argsort(D_low, dim=1)
    nn_high = idx_high[:, 1 : k + 1]
    nn_low = idx_low[:, 1 : k + 1]
    rank_high = torch.argsort(torch.argsort(D_high, dim=1), dim=1)
    rank_low = torch.argsort(torch.argsort(D_low, dim=1), dim=1)

    # 3) Trustworthiness
    U = ~((nn_low.unsqueeze(2) == nn_high.unsqueeze(1)).any(-1))
    impostor_ranks = torch.where(
        U, rank_high.gather(1, nn_low), torch.zeros_like(nn_low)
    )
    penalty_T = torch.clamp(impostor_ranks - k, min=0).sum(dim=1)
    norm_T = k * (2 * n - 3 * k - 1) / 2 if 2 * k < n else (n - k) * (n - k - 1) / 2
    trustworthiness = 1 - penalty_T.float().mean() / (norm_T + eps)

    # 4) Continuity
    V = ~((nn_high.unsqueeze(2) == nn_low.unsqueeze(1)).any(-1))
    missing_ranks = torch.where(
        V, rank_low.gather(1, nn_high), torch.zeros_like(nn_high)
    )
    penalty_C = torch.clamp(missing_ranks - k, min=0).sum(dim=1)
    continuity = 1 - penalty_C.float().mean() / (norm_T + eps)

    # CLEANUP #1: done with idx_* and rank_* and masks/penalties
    del idx_high, idx_low, rank_high, rank_low, U, V
    del impostor_ranks, missing_ranks, penalty_T, penalty_C
    torch.cuda.empty_cache()

    # 5) Scale-Normalized Stress
    D2_high = D_high.pow(2)  # reuse later in Procrustes
    D2_low = D_low.pow(2)
    numerator = (D_high * D_low).sum()
    denominator = D2_low.sum().clamp_min(eps)
    alpha = numerator / denominator
    scale_normalized_stress = (
        (D_high - alpha * D_low) ** 2
    ).sum() / D2_high.sum().clamp_min(eps)

    # CLEANUP #2: done with D2_low, numerator/denominator/alpha
    del D2_low, numerator, denominator, alpha
    torch.cuda.empty_cache()

    # 6) Neighborhood Hit & True Neighbors
    neighborhood_hit = (labels.unsqueeze(1) == labels[nn_low]).float().mean()
    true_neighbors = (
        (nn_low.unsqueeze(2) == nn_high.unsqueeze(1)).any(-1).float().mean()
    )

    # CLEANUP #3: done with nn_low, nn_high for future uses
    del nn_low, nn_high
    torch.cuda.empty_cache()

    # 7) Batched Local Procrustes (TPQM style)
    knn_local = torch.topk(-D2_high, k=k, dim=1)[1]  # (n, k), includes self
    data_neigh = X[knn_local]  # (n, k, d_orig)
    proj_neigh = Y[knn_local]  # (n, k, d_emb)

    proj_mean = proj_neigh.mean(dim=1, keepdim=True)
    proj_centered = proj_neigh - proj_mean
    Z = data_neigh.transpose(1, 2) @ proj_centered
    U_s, S_s, Vh_s = torch.linalg.svd(Z, full_matrices=False)

    trace_sigma = S_s.sum(dim=1)
    denom_scale = proj_neigh.pow(2).sum(dim=(1, 2)).clamp_min(eps)
    c_s = trace_sigma / denom_scale
    A_s = U_s @ Vh_s
    cA = c_s.view(n, 1, 1) * A_s
    mapped = proj_neigh @ cA.transpose(1, 2)
    residual = data_neigh - mapped
    centered_res = residual - residual.mean(dim=1, keepdim=True)
    procrustes_i = centered_res.pow(2).sum(dim=(1, 2))
    data_mean = data_neigh.mean(dim=1, keepdim=True)
    centered_data = data_neigh - data_mean
    denom_p = centered_data.pow(2).sum(dim=(1, 2)).clamp_min(eps)
    procrustes_error = (procrustes_i / denom_p).mean()

    # CLEANUP #4: done with Procrustes intermediates
    del D2_high, knn_local, data_neigh, proj_neigh
    del proj_mean, proj_centered, Z, U_s, S_s, Vh_s
    del trace_sigma, denom_scale, c_s, A_s, cA, mapped
    del residual, centered_res, procrustes_i, data_mean, centered_data, denom_p
    torch.cuda.empty_cache()

    # 8) Pearson Correlation
    iu = torch.triu_indices(n, n, offset=1)
    d_orig_p = D_high[iu[0], iu[1]]
    d_emb_p = D_low[iu[0], iu[1]]  # <- use D_low here
    do = d_orig_p - d_orig_p.mean()
    de = d_emb_p - d_emb_p.mean()  # <- compute de
    pearson_correlation = (do * de).sum() / (do.norm() * de.norm() + eps)

    # CLEANUP #5:
    del D_high, D_low, iu, d_orig_p, d_emb_p, do, de
    torch.cuda.empty_cache()

    # 9) Distance Consistency
    distance_consistency = torch.tensor(float("nan"), device=X.device)
    if labels is not None and n_classes is not None:
        # labels: (n,) ints in [0, n_classes)
        # Y:      (n, d_emb)
        # 1) Sum embeddings per class
        centroids_sum = torch.zeros(n_classes, d_emb, device=Y.device, dtype=Y.dtype)
        centroids_sum = centroids_sum.index_add(
            0, labels, Y
        )  # after this, centroids_sum[c] = sum of Y[i] for all i with labels[i]=c

        # 2) Count points per class
        counts = torch.bincount(labels, minlength=n_classes).clamp_min(1).unsqueeze(1)
        # clamp_min(1) prevents division-by-zero for empty classes

        # 3) Compute centroids
        centroids = centroids_sum / counts  # (n_classes, d_emb)

        # 4) Distances to centroids and nearest
        d2cent = torch.cdist(Y, centroids, p=2)  # (n, n_classes)
        nearest = d2cent.argmin(dim=1)  # (n,)

        # 5) Consistency score
        distance_consistency = (nearest == labels).float().mean()

    return {
        "trustworthiness": trustworthiness.item(),
        "continuity": continuity.item(),
        "scale_normalized_stress": scale_normalized_stress.item(),
        "neighborhood_hit": neighborhood_hit.item(),
        "true_neighbors": true_neighbors.item(),
        "distance_consistency": distance_consistency.item(),
        "procrustes_error": procrustes_error.item(),
        "pearson_correlation": pearson_correlation.item(),
    }
