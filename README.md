# A serving library for various machine learning models

## Background
This library intends to help serving machine learning models in the production environment.

Many models are trained by Python-based machine learning frameworks, which usually can do interfering by themselves.  However, there are several occasions that we prefer to run serve the model not in python but in other frameworks/runtimes, for example:

* The machine learning model is just one component of a big project. and most of the project is not written in Python, but, e.g in Java or Golang. To keep the DevOps work simple, use the runtime/language that the is used by rest part of the project is preferred.

* Running Python-based applications in a production environment is tricky and require a different set of expertises due to the slowness of the VM,  lack of real multitasking, etc. It also usually means more deployments for the same volumn of traffic,  compared to other faster languages/runtimes. Fewer deployments mean less cost.

The library is mainly written in Go, and organized in different folders. Each folder contains a Go package to serve one kind of the machine learning models. 


## The supported machine learning models are:

* LightFM https://github.com/lyst/lightfm
