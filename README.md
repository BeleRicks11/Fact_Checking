# Fact Checking :heavy_check_mark: :heavy_multiplication_x:

In this work we have tried several diﬀerent model architectures to address the **Fact checking task**. First
of all we applied a pre-processing phase cleaning our dataset. Then, starting from the baseline, we
modified the classification head, replaced the Bi-RNNs with a Bi-GRU and performed hyperparameters
tuning on our models. According to the results, the best model seemed to be the one based on the mean
of the Bi-GRU outputs for the sentence embedding part and a double FC layer with a skip connection
(MRNN).

## Authors
* [Riccardo Fava](https://github.com/BeleRicks11)
* [Luca Bompani](https://github.com/Bomps4)
* [Davide Mercanti](https://github.com/nonci)

