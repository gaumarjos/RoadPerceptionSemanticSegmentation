rm -rf optimised_model/*
mkdir optimised_model

~/dev/tf/tensorflow-r1.3/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=frozen_model/saved_model.pb \
--out_graph=optimised_model/saved_model.pb \
--inputs=data/images \
--outputs=predictions/prediction_class \
--transforms='
add_default_attributes
remove_nodes(op=Identity, op=CheckNumerics)
fold_constants(ignore_errors=true)
fold_batch_norms
fold_old_batch_norms
fuse_resize_and_conv
quantize_weights
quantize_nodes
strip_unused_nodes
sort_by_execution_order'
