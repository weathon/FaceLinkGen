# FracFace paper-aligned fixes

This directory began as an exact copy of `methods/fracface` at commit
`33e8e13`. The following changes align the protection transform with
`attacked_papers/md/fracface.md`.

1. **Input range:** Section 3.2 says the RGB input is normalized to `[0, 1]`
   before BDCT. The released caller already supplies `[0, 1]`, so the extra
   `x * 0.5 + 0.5` conversion was removed.
2. **Per-plane pruning:** The three FBA pruning lists are global indices for
   Y, Cb, and Cr. The released DCT code compared all three lists against
   `range(64)`, silently applying only the Y list to every plane. Each list is
   now applied to its own color plane.
3. **Frequency-major FCR:** Section 3.3 and Algorithm 1 partition frequency
   bands with a snake traversal. The retained channels are now arranged
   frequency-major before that traversal instead of plane-major.
4. **Fractal lattice:** Section 3.4 and Appendix A.2 specify randomized secret
   matrices `M0` and `L0`. The released code had no `L0`; the fixed version
   uses it as the relative indexing order of `M0`.
5. **Non-degenerate depth-2 FFM:** The released depth-2 term is a multiple of
   81 and is erased by the later modulo-81 projection. The fixed 9x9 map uses
   the recursive expansion schedule in Appendix Eq. 15, so its outer and
   inner 3x3 levels both affect the selected channel.

The paper is internally inconsistent about the expansion schedule: main-text
Eq. 1 and Algorithm 2 state `3^(2k)`, while Appendix Eq. 15 gives the recursive
product schedule. The fixed method follows Appendix Eq. 15 because the former
degenerates under the paper's modulo-channel projection.

For these attack experiments, a fixed-channel run uses one mapping seeded with
42. A random-channel training run draws one mapping per batch, as specified by
the experiment protocol; evaluation of that checkpoint uses the fixed mapping.
