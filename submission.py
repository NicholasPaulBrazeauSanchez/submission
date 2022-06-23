import numpy as np

class node():
    def __init__(
            self, 
            depth: int,
            n_items: int):
        #Depth of the node wrt the Root. Roots start at 0
        self.depth = depth
        #Number of items in the test set covered by the node
        self.n_items = n_items
        #Average Treatment Effect. The difference between the 
        #Treatment mean and the Control mean of the node's elements
        self.ATE = None
        self.right_child = None
        self.left_child = None
        #Index of the feature we're using as our uphold threshold. None by default
        self.classifier_index = None
        #Numerical threshold a feature needs to meet or exceed to be 
        #classified on the left node
        self.classifier_threshold = None

#for an input array X, create two subarrays, left and right, 
#which are made up of the rows of X. Those rows with the index'th 
#index less than value go into the right array, and the rest go in left
def make_split(X: np.ndarray, index: int, value: float):
    right = X[np.where(X[:,index] > value)]
    left = X[np.where(X[:,index] <= value)]
    return left, right


class UpliftTreeRegressor():
    #Uplift regressor needs Model called in order to fit anything. 
    def __init__(
            self, 
            Max_depth: int = 3,
            Min_samples_leaf: int = 1000,
            Min_samples_leaf_treated: int = 300, 
            Min_samples_leaf_control: int = 300):
        self.Max_depth = Max_depth
        self.Min_samples_leaf = Min_samples_leaf
        self.Min_samples_leaf_treated = Min_samples_leaf_treated
        self.Min_samples_leaf_control = Min_samples_leaf_control
        self.Model = None
    
    # this function accepts a feature matrix with the labels and treatment/control
    # flag as the final elements
    def minClearance(self, split: np.ndarray) -> bool:
        # we may be working with a matrix that is split such that one side has
        # no elements!
        if split.shape[0] == 0: return 
        # do the number of samples in the split meet the minimum
        minSamps = split.shape[0] >= self.Min_samples_leaf
        Treat = split[np.where(split[:,-1] == 1)]
        Control = split[np.where(split[:,-1] == 0)]
        # do the number of treatment flagged samples in the split meet the minimum
        minTest = Treat.shape[0] >=  self.Min_samples_leaf_treated
        # do the number of control flagged samples in the split meet the minimum
        minControl = Control.shape[0] >= self.Min_samples_leaf_control
        return minSamps and minTest and minControl
        
    
    def Build (self, parentNode: node, X: np.ndarray):
        # we calculate the ATE of the node here. We save ourself the 
        # trouble of having to input it for two child nodes and the first
        # root elsewhere in the code. 
        treat = X[np.where(X[:,-1] == 1)]
        control = X[np.where(X[:,-1] == 0)]
        parentNode.ATE = np.average(treat[:,-2]) - np.average(control[:,-2])
        
        if parentNode.depth is self.Max_depth:
            return
        
        # index and val start as None. We'll be using this as an indicator 
        # of whether or not we need to recur
        index = None
        val = None
        maxdelta = 0
        
        # last two indices are reserved for labels and flags!
        for i in range(X.shape[1] - 2):
            column_values = X[:, i]
            unique_values = np.unique(column_values)
            if len(unique_values) > 10:
                percentiles = np.percentile(column_values, 
                                            [3, 5, 10, 20, 30, 50, 70, 80, 90, 
                                             95, 97])
            else:
               percentiles = np.percentile(unique_values, [10, 50, 90])
            threshold_options = np.unique(percentiles) 
            for value in threshold_options:
                left, right = make_split(X, i, value)
                if self.minClearance(left) and self.minClearance(right):
                    # Uplift metric is difference across test and control
                    # within split populations. We want to maximize it across the split
                    leftT = left[np.where(left[:,-1] == 1)]
                    leftC = left[np.where(left[:,-1] == 0)]
                    rightT = right[np.where(right[:,-1] == 1)]
                    rightC = right[np.where(right[:,-1] == 0)]
                    leftDelta = np.average(leftT[:,-2]) - np.average(leftC[:,-2])
                    rightDelta = np.average(rightT[:,-2]) - np.average(rightC[:,-2])
                    delta = abs(leftDelta - rightDelta)
                    if delta > maxdelta:
                        maxdelta = delta 
                        index = i
                        val = value
        # if index isn't equal to none, this implies that we've found a split
        # for our data and need to recur. Otherwise we haven't
        if index != None:
            # it's more memory efficient to simply store the index and val variables
            # that can help us regenerate a split rather than the entire data 
            # for the split
            leftData, rightData = make_split(X, index, val)
            Left = node(parentNode.depth + 1, leftData.shape[0])
            Right = node(parentNode.depth + 1, rightData.shape[0])
            parentNode.right_child = Right
            parentNode.left_child = Left
            parentNode.classifier_index = index 
            parentNode.classifier_threshold = val
            self.Build(Left, leftData)
            self.Build(Right, rightData)
                
                
                
                
        
    
    def fit(
            self, 
            X: np.ndarray,
            Treatment: np.ndarray,
            Y: np.ndarray):
        Root = node(0, X.shape[1])
        # When fitting our model, we append both our label and our treatment
        # variable to the end of our input variable, to make this data easier 
        # to parse
        summaryMatrix = np.hstack((X, Y[:, np.newaxis], 
                                   Treatment[:, np.newaxis]))
        
        self.Build(Root, summaryMatrix)
        self.Model = Root
        
    def getPred(self, i: np.ndarray, thisNode: node) -> float:
        if thisNode.classifier_index is None:
            return thisNode.ATE
        else:
            if(i[thisNode.classifier_index] <= thisNode.classifier_threshold):
                return self.getPred(i, thisNode.left_child)
            else:
                return self.getPred(i, thisNode.right_child)
        
        
    # we're going to make the assumption that the ATE of a node is the mean-squared
    # estimate for the ATE of a datapoint matched to that group
    def predict(self, X: np.ndarray) -> list():
        if self.Model == None:
            raise Exception("Model not yet fit")
        predictions = []
        for i in X:
            predictions.append(self.getPred(i, self.Model))
        return predictions

def main():
    # Here we run our code on the given test .npy files to make sure everything works
    # end to end. As I've run it, it's all good!
    # note that this requires that submission.py be in the same directory as 
    # Example_x.npy etc
    tree = UpliftTreeRegressor(3, 6000, 2500, 2500)
    X = np.load("Example_x.npy")
    Treatment = np.load("Example_treatment.npy")
    Y = np.load("Example_y.npy")
    tree.fit(X, Treatment, Y)
    GoldPredictions = np.load("example_preds.npy")
    MyPredictions = tree.predict(X)
    np.testing.assert_array_almost_equal(GoldPredictions, np.array(MyPredictions))

if __name__ == "__main__":
    main()