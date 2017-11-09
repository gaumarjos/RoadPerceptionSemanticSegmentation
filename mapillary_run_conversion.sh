rm ../mapillary/data/train/*_gt.png
rm ../mapillary/data/train/*_image.png
rm ../mapillary/data/train/*_stats.png
python mapillary_convert_labels_to_cityscapes_format.py 

