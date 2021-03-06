======================
NVIDIA GPU Cloud (NGC)
======================

| https://ngc.nvidia.com/catalog/collections/nvidia:gettingstarted 
| https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html 
| https://docs.nvidia.com/ngc/index.html 
| https://ngc.nvidia.com/catalog 
| https://www.nvidia.com/en-us/gpu-cloud/ 

The NGC Catalog is a curated set of GPU-optimized software for AI, HPC and Visualization. It consists of containers, pre-trained models, Helm charts for Kubernetes deployments and industry specific AI toolkits with software development kits (SDKs). 

Pre-trained models
==================

| https://ngc.nvidia.com/catalog 
| https://ngc.nvidia.com/catalog/models?orderBy=scoreDESC&pageNumber=0&query=clara_pt&quickFilter=&filters= 

Models that worked:
===================

* clara_pt_brain_mri_annotation_t1c
* clara_pt_brain_mri_segmentation_t1c
* clara_pt_covid19_ct_lesion_segmentation
* clara_pt_covid19_ct_lung_annotation
* clara_pt_covid19_ct_lung_segmentation
* clara_pt_deepgrow_2d_annotation
* clara_pt_deepgrow_3d_annotation
* clara_pt_liver_and_tumor_ct_segmentation
* clara_pt_pancreas_and_tumor_ct_segmentation
* clara_pt_prostate_mri_segmentation
* clara_pt_spleen_ct_annotation
* clara_pt_spleen_ct_segmentation


Datasets
========

https://www.synapse.org/#!Synapse:syn3193805/wiki/217789 


Download Models
===============

Set :code:`AIAA_PORT` to the port of your AIAA server::

   AIAA_PORT=5000

Set the MODEL variable to the name of the new model. For example, for the model :code:`clara_pt_covid19_ct_lung_annotation`::

   MODEL='clara_pt_covid19_ct_lung_annotation' && VERSION=1

Download::

   curl -X PUT "http://127.0.0.1:$AIAA_PORT/admin/model/$MODEL" -H "accept: application/json" -H "Content-Type: application/json" -d '{"path":"nvidia/med/'"$MODEL"'","version":"'"$VERSION"'"}' 