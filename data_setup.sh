#!/bin/bash

pip install pycocotools

## this is dumb, there is a pip package already
#(cd /opt; git clone https://github.com/cocodataset/cocoapi.git)
#(cd /opt/cocoapi/PythonAPI; make )

#(cd /opt/cocoapi; wget -nc http://images.cocodataset.org/annotations/annotations_trainval2014.zip; unzip -n annotations_trainval2014.zip ) &
#(cd /opt/cocoapi; wget -nc http://images.cocodataset.org/zips/train2014.zip; unzip -n train2014.zip -d images/train2014 ) &
(cd /opt/cocoapi; wget -nc http://images.cocodataset.org/zips/val2014.zip; unzip -n val2014.zip -d images/val2014 ) &
(cd /opt/cocoapi; wget -nc http://images.cocodataset.org/zips/test2014.zip; unzip -n test2014.zip -d images/test2014 ) &
(cd /opt/cocoapi; wget -nc http://images.cocodataset.org/zips/image_info_test2014.zip; unzip -n image_info_test2014.zip -d images/image_info_test2014 ) &
