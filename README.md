# A serving library for various machine learning models

This library intends to help serving machine learning models, which are usually created by python based machine learning frameworks ( lightfm, scikit learn, etc), in the production environment.

Python-based machine learning frameworks usually can do interfering by themselves.  But there are several reasons that people don't want to run Python-based applications in production environments.

* The machine learning model is just one component of a big project. and most of the project is not written in Python but Java or Golang. To keep the DevOps work simple,  there is a good reason to use the same language and framework for machine learning models interfering.
* Running Python-based applications in a production environment is actually tricky due to the slowness of the VM,  lack of real multitasking, etc. It usually means more deployments for the same amount of traffic,  compared to other languages/runtimes,  e.g. Go lang or Java.  fewer deployments mean less cost.

The library is mainly written in Go, and organized in different folders. Each folder contains a Go package to serve one kind of the machine learning models. The supported machine learning models are:

* LightFM https://github.com/lyst/lightfm
