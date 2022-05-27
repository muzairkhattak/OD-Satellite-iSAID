# Object Detection in Aerial Images: A Case Study on Performance Improvement

Object Detection (OD) in aerial images has gained much attention due to its applications in search and rescue, town planning, and agriculture yield prediction etc. Recently introduced large-scale aerial images dataset, iSAID has enabled the researchers to advance the OD tasks on satellite images. Unfortunately, the available OD pipelines and ready-to-train architectures are well-tailored and configured to be used with tasks dealing with natural images. In this work, we study that directly using the available object detectors, specifically the vanilla Faster RCNN with FPN is sub-optimal for aerial OD. To help improve its performance, we tailor the Faster R-CNN architecture and propose several modifications including changes in architecture in different blocks of detector, training \& transfer learning strategies, loss formulations, and other pre-post processing techniques. By adopting the proposed modifications on top of the vanilla Faster-RCNN, we push the performance of the model and achieve an absolute gain of 4.44 AP over the vanilla Faster R-CNN on the iSAID validation set.


## Technical Report 
Complete technical report can be viewed [here](https://github.com/MUKhattak/OD-Satellite-iSAID/blob/OD_SatteliteImages/projects/OD_satellite_iSAID/technical_report.pdf).
