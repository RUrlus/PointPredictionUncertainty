# Point Prediction Uncertainty

At ING-bank we tend to evaluate machine learning models based on group-level metrics such as precision-recall, F1 or AUCROC, evaluated on an entire dataset.
However, in many business settings knowing the prediction uncertainty on individual data points is critical, as a model should refrain from predicting when there is high uncertainty. 
You will work on an innovative method to find the prediction uncertainty on individual data points, and try to apply this to some models used at ING.

## Problem statement

Let $S = \{(\mathbf{x}_1, y_1), \dots, (\mathbf{{x}}_n, y_n)\}$ be a set of $n$ i.i.d. (input, target) pairs that follow an unknown distribution $\mathcal{P}$.
Where $y_1, \cdots, y_n \in \{0, \cdots, k\} \subset \mathbb{Z}$ and $\mathbf{x}_1, \cdots, \mathbf{x}_n \in \mathbb{R}^{d}; \mathbf{x}_1, \cdots, \mathbf{x}_n \sim \mathcal{P}_{\mathbf{x}}$, where $\mathcal{P}_{\mathbf{x}}$ is the marginal distribution of the $\mathbf{x}$.

<!--$P:\mathbb{R}^d \times \{0, 1\} \to [0, 1]$ that maximises (minimises) the target metric $e$: -->

Suppose we have a set of predictor which is called *Hypothesis class*. Our goal is to learn a function $h \in \mathcal{H}; h: \mathbb{R}^{d} \to \{0, \cdots, k\}$ that maximises (minimises) the target metric $e$: $\operatorname{max}_{h}\mathbb{E}[e(h(x), y)]$. A properly calibrated model predicts $P(y_i = C_j | \mathbf{x}_i)$, the probability that observation $y_i$ belongs to class $C_j$.


Yet a model can predict a score of 0.5 for an observation with very low uncertainty, meaning the model is very sure the probability of either class is equal.
Existing model confidence scores are based on evaluating a hold-out set over (many) permutations of the classifier.
Evaluating the variability of the predicted class for these points over the permutations gives a measure of uncertainty.
Various flavours of model confidence scores exist[@mandelbaum2017;@lakshminarayanan2017;@gruber2023].


However, all these methods fail in one regard: the further an observation lies from the decision boundary the higher the model confidence tends to be.
This confidence may not be warranted if the model has seen a few or even no observations for a particular region.


Many machine learning models are essentially curve fitters which can exhibit erratic behaviour for out-of-domain observations.
This problem quickly becomes worse in high dimensional feature spaces which are often heterogeneous in density and can result in highly non-linear decision boundaries. 
What is needed is a unified measure of uncertainty that not only incorporates the model’s uncertainty but also the statistical uncertainty inherent to having only a limited sample.


To summarise, let $\mathcal{X} = \{ \mathbf{x}_{n+1}, \cdots, \mathbf{x}_{m} \}$, where $\mathbf{x}_{n+1}, \cdots, \mathbf{x}_{m} \sim \mathcal{P}_{\mathbf{x}}$ and $\mathcal{X} \cap X = \emptyset$, be a set of inputs for which we want to estimate the corresponding target.
For a given observation $\mathbf{x} \in \mathcal{X}$ there will be the uncertainty for the result of the classifier, which we can divided it into mainly 2 parts, the out of distribution classifier uncertianty and training uncertainty:


* $\sigma_{\mathrm{OODC}}$: scince we trained the classifier on the training set, so the training set is all the information we know about the data distribution, while the situation that the new data(test set) from the distribution looks like an outlier for the training set, so it's like the probability that the feature of the new data point is not captured by the training set.

$$
\begin{aligned}
\sigma_{\text{OODC}} &= Prob(\text{the $\mathbf{x}$ acts like a data point which we have not seen before})\\
&= Prob(\text{the training set is bad} \mid \mathbf{x})
\end{aligned}
$$


* $\sigma_{\mathrm{training}}$:When we have a learning algorithm, in real life cases, we are going to train the model on a training set, while the extant of the training set capturing the feature of the distribution has variance, and the training environments, including software parameter initialization, which leads to the variance of the point predictions by models. This is the uncertainty of the classifier generating process itself. Given a learning algorithm and given a set of training sets and training environments, it can generate a hypothesis class $\mathcal{H}$, where 

$$
\sigma_{\mathrm{training}} = var(h(x) \mid h \in \mathcal{H})
$$

## Out of Distribution Classifier Uncertianty

### Kullback-Leibler divergence

For the given training set $S_n = \{(\mathbf{x}_1, y_1), \dots, (\mathbf{{x}}_n, y_n)\}$, we can estimate the joint probability distribution of $(\mathbf{x}, y)$, which is $\hat{P}_{S_n}(\mathbf{x}, y) = \hat{P}_n(y\mid\mathbf{x}) \cdot \hat{P}_{\mathbf{X}_n}(\mathbf{x})$, where $\mathbf{X}_n = \{ \mathbf{x}_1, \cdots, \mathbf{x}_n \}$. If we got a new data point $\mathbf{x}_{n+1}$ without the label, we have $\mathbf{X}_{n+1} = \{ \mathbf{x}_1, \cdots, \mathbf{x}_{n+1} \}$, then we can estimate it again $\hat{P}_{S_n,\mathbf{x}_{n+1}}(\mathbf{x}, y) = \hat{P}_{n}(y\mid\mathbf{x}) \cdot \hat{P}_{\mathbf{X}_{n+1}}(\mathbf{x})$, then we can calculate the  KL divergence between them.

$$
\begin{align*}
    &KL(\hat{P}_{S_n, \mathbf{x}_{n+1}};\hat{P}_{S_n})\\ =& \int \log\left( \frac{\hat{P}_{S_n, \mathbf{x}_{n+1}}(\mathbf{x}, y)}{\hat{P}_{S_n}(\mathbf{x}, y)}\right)\hat{P}_{S_n, \mathbf{x}_{n+1}}(\mathbf{x}, y)d\mathbf{x}dy \\
    =& \int \log\left( \frac{\hat{P}_{n}(y\mid\mathbf{x}) \cdot \hat{P}_{\mathbf{X}_{n+1}}(\mathbf{x})}{\hat{P}_{n}(y\mid\mathbf{x}) \cdot \hat{P}_{\mathbf{X}_n}(\mathbf{x})}\right) \hat{P}_{n}(y\mid\mathbf{x}) \cdot \hat{P}_{\mathbf{X}_{n+1}}(\mathbf{x})d\mathbf{x}dy\\
    =& \int \log \left( \frac{\hat{P}_{n}(y \mid \mathbf{x})}{\hat{P}_n(y \mid \mathbf{x})} \right) \hat{P}_{n}(y\mid\mathbf{x}) \cdot \hat{P}_{\mathbf{X}_{n+1}}(\mathbf{x})d\mathbf{x}dy \\
    &+ \int \log \frac{\hat{P}_{\mathbf{X}_{n+1}}(\mathbf{x})}{\hat{P}_{\mathbf{X}_n}(\mathbf{x})}  \hat{P}_{n}(y\mid\mathbf{x}) \cdot \hat{P}_{\mathbf{X}_{n+1}}(\mathbf{x})d\mathbf{x}dy\\
    =& KL(\hat{P}_{\mathbf{X}_{n+1}};\hat{P}_{\mathbf{X}_n})
\end{align*}
$$

 which shows that the Kullback-Leibler divergence between two estimated $\mathbf{x}$ densities, just refleces the Kullback-Leibler divergence between two estimated joint densities. We can then use this to estimate how bad we capture the real joint distribution after we have seen $\mathbf{x}$.

## Training Uncertainty

### Bregman Information

We will first introduce the Bregman divergence, the  **Bregman divergence** generated by a diffrentiable, convex function $\phi:U \to \mathbb{R}$, where $U$ is a convex set, is defined as

$$
d_{\phi}(x, y) = \phi(y) - \phi(x) - \langle\nabla \phi (x), y - x\rangle
$$

which can be interpreted geometrically as the difference between $\phi$ and the supporting tangent plane of $\phi(x)$ at $y$.

The **Bregman Information**(generate by $\phi$) of a random variable $X$ with realizations in $U$ is defined as

$$
\begin{align}
\mathbb{B}_{\phi}[X] &= \mathbb{E}\left[d_{\phi}\left(\mathbb{E}[X], X\right)\right]\\
&=\mathbb{E}[\phi(X) - \phi(\mathbb{E}[X]) - \langle\nabla \phi (\mathbb{E}[X]), x - \mathbb{E}[X]\rangle]\\
&=\mathbb{E}[\phi(X)] - \phi(\mathbb{E}[X])
\end{align}
$$

which is the generalizes the variance of a random variable since take $U = \mathbb{R}$ and $\phi(x) = x^2$.

For logit prediction $\hat{z} \in \mathbb{R}^k$ and target $Y \sim Q$ with $k$ classes, we have

$$
\mathbb{E}[-\ln sm_Y(\hat{z})] = H(Q) + \mathbb{B}_{LSE}[\hat{z}] + d_{LSE}(sm^{-1}(Q),\mathbb{E}[\hat{z}])
$$

with the *LogSumExp Function* $LSE(x_1,\cdots,x_n) = \ln\sum^n_{i=1}e^{x_i}$, the *softmax function* $sm = \nabla LSE$ and *Shannon entropy* $H$[@gruber2022uncertainty]. 

### Score Variance


## References

