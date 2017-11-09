rm ../mapillary/data/training/images/*_cropped.png
rm ../mapillary/data/training/instances/*_cropped.png
python mapillary_convert_labels_to_cityscapes_format.py
