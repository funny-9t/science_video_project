# Aesthetic Prompt Feature Sanity Check

- Videos checked in the random table: 20 (seed=42).
- Full feature population: 615.
- A2 cache metadata valid: 615/615.
- NaN/Inf: A1=0, A2=0.
- Per-dimension standard deviation A1: `[0.011141814291477203, 0.016562076285481453, 0.0070504057221114635, 0.013003069907426834, 0.011487147770822048, 0.005391020327806473, 0.017745941877365112]`.
- Per-dimension standard deviation A2: `[0.00779994810000062, 0.009499378502368927, 0.010467219166457653, 0.007383172865957022, 0.00880134291946888, 0.009519360028207302, 0.008408354595303535]`.
- Near-constant dimensions (std < 1e-6): A1=0, A2=0.
- Maximum absolute within-set off-diagonal Pearson correlation: A1=0.458887, A2=0.747365.
- Mean absolute value: A1=0.027545, A2=0.013540.
- Global standard deviation: A1=0.032253, A2=0.013913.
- Deterministic repeat max absolute delta for `7629782677985529088`: 0.0000000000.
- Aggregation remains frame-wise mean; no sigmoid, softmax, or normalization was added after the CLIP similarity difference.
