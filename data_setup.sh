!pip install pycocotools
!(cd /opt; git clone https://github.com/cocodataset/cocoapi.git)
!(cd /opt/cocoapi; wget -nc http://images.cocodataset.org/annotations/annotations_trainval2014.zip; unzip -n annotations_trainval2014.zip )
!(cd /opt/cocoapi; wget -nc http://images.cocodataset.org/zips/train2014.zip; unzip -n train2014.zip )
!(cd /opt/cocoapi; wget -nc http://images.cocodataset.org/zips/val2014.zip; unzip -n val2014.zip )
!(cd /opt/cocoapi; wget -nc http://images.cocodataset.org/zips/test2014.zip; unzip -n test2014.zip )