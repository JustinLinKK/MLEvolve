We are design a model dataset for testing our job scheduler. And my teammate give me this 3 files for the timeline list generation. However, there was no model source code given so we have to fully rework the whole dataset generation thing.

We need a list of 100 Job list (we call it A) with each job correpsonding model source code, each job's epoches would be 50, each model should have some structure/datatype variance and the model base structure should also be vary (CNN,MLP,LSTM,etc). We would use dataset histopathologic-cancer-detection so carefully design your model and test them on that dataset for one epoches to verify that it works.

Then we need to generate a poission distribution timeline using A, the timeline should be documented so we could precisely replay that timeline trace on a benchmark test

Then in the benchmark test we would submit training jobs based on that timeline file and measure the metrics over that.