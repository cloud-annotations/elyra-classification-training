# Title: How to build a scalable open source (Trusted AI) Visual Recognition pipeline with Elyra, TensorFlow and Kubernetes

# Introduction
This article provides a entirely open source, reusable template for production grade computer vision (image classification and annotation) on top of Kubernetes. It is meant for data scientists, data engineers, and AI/data centric software engineers. You will learn how to use the [Cloud Annotations](https://github.com/cloud-annotations) tool, [Kubeflow Pipelines](https://kubeflow.org), the TrustedAI toolkits [AIF360](https://github.com/Trusted-AI/AIF360) and [AIX360](https://github.com/Trusted-AI/AIX360) on top of [Elyra](https://github.com/elyra-ai), making [Kubernetes](https://kubernetes.io/) as first class citizen of [JupyterLab](https://jupyter.org)

We visually create an AI pipeline with a set of jupyter notebooks. We cover two workflows. First, with your own image dataset and the Cloud Annotation tool you label your favorite images into categories, upload it to S3 cloud object store and have a ready made pipeline classify the images for you and deploy this visual recognition model as REST service to Kubernetes. Then, using the "Fairface" dataset we train and thenn asses the trained classification model on fairness metrics like bias using the open source AIF360 toolkit. We will detect if images from underprivileged group are experiencing reduced model performance. Finally, we use the AIX360 toolkit to highlight parts of the images which have been crucial for the classifiers decision.
    
    
# Prerequisites
    List or describe any skills, tools, experience, or specific conditions required to understand the article. Include version levels for any required tools or platforms. Include links to necessary resources whenever possible.
# Estimated time
    Provide guidance on how long it will reasonably take to read the article.
# Summary
    State any closing remarks about the concept you described and its importance. Recommend a next step (with link if possible) where they can continue to expand their skills after completing your article.
# Related links
    Include links to other resources that may be of interest to someone who is reading your article.
