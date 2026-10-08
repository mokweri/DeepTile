# DeepTile C++ API {#mainpage}

This reference is generated from the public C++ headers in `src/`, following
the repository's comment style guide. Browse by subsystem:

- @ref deeptile_common — architecture constants, packed descriptors, and the
  versioned host-to-kernel contract.
- @ref deeptile_compiler — ModelPackage loading, graph lowering, scheduling,
  and descriptor emission.
- @ref deeptile_host — runtime planning, XRT execution, dataset evaluation, and
  RunRecords.
- @ref deeptile_kernel — the HLS kernel entry point and streaming datapath.

The usual software path begins at deeptile::graph::compile_graph(), produces a
deeptile::runtime::RuntimeExecutionPlan, and executes it through
deeptile::runtime::XrtBackend. The hardware entry point is tile_conv().

For concepts, guides, and specifications, return to the
[documentation site](../../index.html).
