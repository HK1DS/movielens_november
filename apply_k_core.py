import pandas as pd
import numpy as np
import argparse
import os

def filter_k_core(df, k):
    """Iteratively filter dataframe to obtain the (k, k)-core (rows = interactions).

    Works on integer category codes with a boolean row mask, so a 10M-row file
    needs a few hundred MB instead of several GB. The k-core is the unique
    maximal fixed point, so the kept rows (and their original order) are
    identical to the previous pandas value_counts/isin implementation.
    """
    u = df['user_id'].astype('category').cat.codes.to_numpy()
    i = df['item_id'].astype('category').cat.codes.to_numpy()
    nu, ni = int(u.max()) + 1, int(i.max()) + 1
    keep = np.ones(len(df), dtype=bool)
    iteration = 0
    while True:
        before = int(keep.sum())
        ucnt = np.bincount(u[keep], minlength=nu)          # user core filter
        keep &= ucnt[u] >= k
        icnt = np.bincount(i[keep], minlength=ni)          # item core filter
        keep &= icnt[i] >= k
        after = int(keep.sum())
        iteration += 1
        print(f"Iteration {iteration}: Users {len(np.unique(u[keep]))}, "
              f"Items {len(np.unique(i[keep]))}, Rows {after}", flush=True)
        if after == before:
            break
    return df[keep]


def main():
    parser = argparse.ArgumentParser(description="Iterative (k, k)-core filtering of an interactions CSV.")
    parser.add_argument("--interactions_in", default="data/interactions.csv", help="user_id,item_id,rating,timestamp")
    parser.add_argument("--k", type=int, default=20, help="k value for k-core filtering (default: 20)")
    parser.add_argument("--out_dir", default="data/k_core", help="Output directory for filtered files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not os.path.exists(args.interactions_in):
        raise SystemExit(f"Error: '{args.interactions_in}' not found. Run prepare_interactions.py first.")

    # category dtype keeps each distinct id string once (memory-safe for ~10M+ rows)
    df_inter = pd.read_csv(args.interactions_in, dtype={"user_id": "category", "item_id": "category"})
    print(f"Original interactions: {df_inter.shape}")
    n_dup = int(df_inter.duplicated(["user_id", "item_id"]).sum())
    if n_dup:
        print(f"[warn] {n_dup:,} repeated (user, item) pairs. RecBole keeps them as separate rows; "
              f"decide in the adapter whether re-interactions should be kept.")

    print(f"\nFiltering interactions with k={args.k}...")
    filtered_inter = filter_k_core(df_inter, args.k)
    n_u = filtered_inter["user_id"].nunique()
    n_i = filtered_inter["item_id"].nunique()
    print(f"k-core: {n_u} users, {n_i} items, {len(filtered_inter)} interactions")

    interactions_out = os.path.join(args.out_dir, f"interactions_k{args.k}.csv")
    filtered_inter.to_csv(interactions_out, index=False)
    print(f"[Saved] interactions -> {interactions_out}")


if __name__ == "__main__":
    main()
