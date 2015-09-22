
## Distribution Drift

Say you train a model on some historical data and then deploy your model in a production setting where it is working with live data.

It is possible that the distribution of the live data starts to _drift_ from the distribution your model learned. This change may be due to factors in the real world that influence the data.

Ideally you will be able to detect this drift as it happens, so you know whether or not your model needs adjusting. A simple way to do it is to continually evaluate the model by computing some validation metric on the live data. If the distribution is stable, then this validation metric should remain stable; if the distribution drifts, the model starts to become a poor fit for the new incoming data, and the validation metric will worsen.


## References

- Evaluating Machine Learning Models. Alice Zheng. 2015.
