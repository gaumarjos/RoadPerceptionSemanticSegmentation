# first parameter $1 frozen_model_dir

~/dev/tf/tensorflow-r1.3/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=$1/graph.pb \
--out_graph=$1/optimised_graph.pb \
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
