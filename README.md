# machine-learning-note
This is a note for machine learning

# Gradient Boosting Decision Tree (GBDT)
### XgBoost
  - Pre-sorting:
    - For each node, enumerate over all features
    - For each feature, sort the instances by feature value
    - Use linear scan to decide the best split along that feature basis information gain.
    - Take the best split solution along all the features
  - Histogram-based:
    - Split all data points for a feature into discrete bins
    - Uses these bins to find the best split value of histogram

- Feature importance
  - weight/frequency: the number of time a feature is used to split the data across all trees.

    For example: feature#1 used 2, 3, 4 times to split nodes in tree#1, tree#2 and tree#3 respectively, feature#1 weight is $2+3+4=9$. Using this feature importance may have impact on binary features or features with less possible values, i.e. those features may have lower weight importance than features with mass possible values.The frequency for feature#1 is calculated as its percentage weight over weights of all features.

  - gain: The average information gain across all splits the feature is used in.

    For example: 
    ![](https://cdn.mathpix.com/snip/images/U9w5dEqgj1zand9NNrFkPF-9kc8O6NU-dWs-pFixKoA.original.fullsize.png)

    In summary:
    - step 1: calculate previous node information entropy
    - step 2: calculate all child nodes information entropy and get weighted sum
    - step 3: differentia these two information entropy and weight it according the proportion of current samples affected by the split(in the example it is all samples)
    - step 4: sum up across all trees and splits the feature is used in

  - cover: The average coverage across all splits the feature is used in

    For example: feature#1 split 10, 5, 2 samples in tree#1, tree#2 and tree#3 respectively, feature#1 weight is $10+5+2=17$. This will be calculated for all features as a percentage for all features' cover metrics.

### LightGBM
  - Gradient-based One-Side Sampling (GOSS)
    - Keeps all the instances with large gradients and performs random sampling on the instances with small gradients
    - Training instances with small gradients have smaller training error and it is already well-trained.
    - To achieve good balance between reducing the number of data instances and keeping the accuracy for learned decision trees, GOSS introduces a constant multiplier for the data instances with small gradients.

- Feature importance
  - split: (identical to xgboost weight/frequency) The number of times the feature is used in the model
  - gain: (identical to xgboost gain) Total gains of splits when feature is used

### CatBoost
  - Practical implement of orderd boosting
  - Permutating the set of input observations in a random order, multiple random permutations are generated
  - Converting the label value from a floating point or category to an integer
  - Transform all categorical feature to numeric values using avg\_target=(countInClass+prior)/(totalCount+1)    
    - countInClass is how many times the label value was equal to "1" for objects with the current categorical feature value.
    - TotalCount is the total number of objects that have a categorical feature value matching the current one

![](/images/Picture1.png)

- Feature importance
  - predicted value changes: How munch on average the prediction changes if the feature value changes. The bigger the value of the importance, the bigger on average is the change to the prediction value.
  - loss function change: Represents the difference between the loss value of the model with this feature and without it. The model without the feature is equivalent to the one that would have been trained if this feature was excluded from the dataset. Since it is computationally expensive to retrain the model without one of the features, this model is built approximately using the original model with this feature removed from all the trees in the ensemble.
  - internal feature importance: Importance of values both for each of the input features and their combinations.
  - prediction difference: Impact of a feature on the prediction results for a pair of objects. For each feature PredictionDiff reflects the maximum possible change in the predictions difference if the values of the feature is changed for both objects. The change is considered only if it affects the ranking order of objects.

