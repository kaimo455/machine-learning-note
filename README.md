# machine-learning-note
This is a note for machine learning

### XgBoost
  - Pre-sorting:
    - For each node, enumerate over all features
    - For each feature, sort the instances by feature value
    - Use linear scan to decide the best split along that feature basis information gain.
    - Take the best split solution along all the features
  - Histogram-based:
    - Split all data points for a feature into discrete bins
    - Uses these bins to find the best split value of histogram

### LightGBM
  - Gradient-based One-Side Sampling (GOSS)
    - Keeps all the instances with large gradients and performs random sampling on the instances with small gradients
    - Training instances with small gradients have smaller training error and it is already well-trained.
    - To achieve good balance between reducing the number of data instances and keeping the accuracy for learned decision trees, GOSS introduces a constant multiplier for the data instances with small gradients.

### CatBoost
  - Practical implement of orderd boosting
  - Permutating the set of input observations in a random order, multiple random permutations are generated
  - Converting the label value from a floating point or category to an integer
  - Transform all categorical feature to numeric values using avg\_target=(countInClass+prior)/(totalCount+1)    
    - countInClass is how many times the label value was equal to "1" for objects with the current categorical feature value.
    - TotalCount is the total number of objects that have a categorical feature value matching the current one

![](./images/Picture1.png)
