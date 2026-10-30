# Frozen inference source

`uncertainty_v1_source.zip` preserves the exact 34 source files named by the
selected uncertainty model's source map. It was extracted from the verified
`bob_cyclone_uncertainty_code.zip` source bundle whose SHA-256 is
`9e86981d2341a7b35b023086d267e8b00e40b02219ce7589a1ecba44f5aa9647`.

The snapshot ZIP's SHA-256 is
`ffe8964d5a43194f7a62708a930c07c644d99cfa76cac204a50e6467cbe50051`.
The canonical source-map digest is
`56a9bd2d120cd67664de3178bf9165baaa6da681274a0b515276a98f66dc8503`.

The final evaluator verifies every file and uses this snapshot in a separate
inference worker. The original source checks remain active. The current project
supplies the new evaluation/scoring code and the immutable data packages; the
selected calibrated model is supplied separately in its reviewed results ZIP.
