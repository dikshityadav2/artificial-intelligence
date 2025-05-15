from approvedimports import *

def make_xor_reliability_plot(train_x, train_y):
    """ Insert code below to complete this cell according to the instructions in the activity descriptor.
    Finally it should return the fig and axs objects of the plots created.

    Parameters:
    -----------
    train_x: numpy.ndarray
        feature values

    train_y: numpy array
        labels

    Returns:
    --------
    fig: matplotlib.figure.Figure
        figure object

    ax: matplotlib.axes.Axes
        axis
    """

    # ====> insert your code below here
    # Create a list of hidden layer sizes from 1 to 10
    hidden_layer_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Lists to store results
    success_counts = [0] * 10  # How many times each size gets 100% accuracy
    epoch_list = [[0 for _ in range(10)] for _ in range(10)]  # Store epochs for each run

    # Loop through each hidden layer size
    for size_index in range(len(hidden_layer_sizes)):
        current_size = hidden_layer_sizes[size_index]

        # Try 10 times for this size
        for trial in range(10):
            # Create a new MLP model
            mlp_model = MLPClassifier(
                hidden_layer_sizes=(current_size,),
                max_iter=1000,
                alpha=0.0001,
                solver="sgd",
                learning_rate_init=0.1,
                random_state=trial
            )

            # Train the model with the data
            mlp_model.fit(train_x, train_y)

            # Check how accurate it is
            accuracy = mlp_model.score(train_x, train_y) * 100

            # If accuracy is 100%, count it and save the epochs
            if accuracy == 100:
                success_counts[size_index] = success_counts[size_index] + 1
                epoch_list[size_index][trial] = mlp_model.n_iter_

    # Calculate average epochs for successful runs
    average_epochs = [0] * 10
    for size_index in range(10):
        total_epochs = 0
        successful_runs = 0

        # Check each trial for this size
        for trial in range(10):
            if epoch_list[size_index][trial] > 0:
                total_epochs = total_epochs + epoch_list[size_index][trial]
                successful_runs = successful_runs + 1

        # Compute average or set to 1000 if no successes
        if successful_runs > 0:
            average_epochs[size_index] = total_epochs / successful_runs
        else:
            average_epochs[size_index] = 1000

    # Create the plots
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left plot: show how many successes
    axes[0].plot(hidden_layer_sizes, success_counts, marker='o')
    axes[0].set_title("Reliability")
    axes[0].set_xlabel("Hidden Layer Width")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_xticks(hidden_layer_sizes)

    # Right plot: show average epochs
    axes[1].plot(hidden_layer_sizes, average_epochs, marker='o')
    axes[1].set_title("Efficiency")
    axes[1].set_xlabel("Hidden Layer Width")
    axes[1].set_ylabel("Mean Epochs")
    axes[1].set_xticks(hidden_layer_sizes)

    # Make the plots look nice
    plt.tight_layout()
    # <==== insert your code above here

    return figure, axes

# make sure you have the packages needed
from approvedimports import *

#this is the class to complete where indicated
class MLComparisonWorkflow:
    """ class to implement a basic comparison of supervised learning algorithms on a dataset """

    def __init__(self, datafilename:str, labelfilename:str):
        """ Method to load the feature data and labels from files with given names,
        and store them in arrays called data_x and data_y.

        You may assume that the features in the input examples are all continuous variables
        and that the labels are categorical, encoded by integers.
        The two files should have the same number of rows.
        Each row corresponding to the feature values and label
        for a specific training item.
        """
        # Define the dictionaries to store the models, and the best performing model/index for each algorithm
        self.stored_models:dict = {"KNN":[], "DecisionTree":[], "MLP":[]}
        self.best_model_index:dict = {"KNN":0, "DecisionTree":0, "MLP":0}
        self.best_accuracy:dict = {"KNN":0, "DecisionTree":0, "MLP":0}

        # Load the data and labels
        # ====> insert your code below here
        # Let's load the files here!
        self.data_x = np.genfromtxt(datafilename, delimiter=",")
        self.data_y = np.genfromtxt(labelfilename, delimiter=",")
        # <==== insert your code above here

    def preprocess(self):
        """ Method to
           - separate it into train and test splits (using a 70:30 division)
           - apply the preprocessing you think suitable to the data
           - create one-hot versions of the labels for the MLP if there are more than 2 classes

           Remember to set random_state = 12345 if you use train_test_split()
        """
        # ====> insert your code below here
        # Split data into 70% train, 30% test
        sample_count = len(self.data_y)
        train_portion = int(0.7 * sample_count)

        # Use train_test_split for splitting
        train_features, test_features, train_classes, test_classes = train_test_split(
            self.data_x, self.data_y, test_size=0.3, stratify=self.data_y, random_state=12345
        )
        self.train_x = train_features
        self.test_x = test_features
        self.train_y = train_classes
        self.test_y = test_classes

        # Normalize features to range 0-1
        feature_count = len(self.train_x[0])
        train_normalized_features = []
        test_normalized_features = []

        for sample in self.train_x:
            normalized_row = []
            for j in range(feature_count):
                column_data = [row[j] for row in self.data_x]
                min_val = min(column_data)
                max_val = max(column_data)
                if max_val == min_val:
                    scaled_value = 0
                else:
                    scaled_value = (sample[j] - min_val) / (max_val - min_val)
                normalized_row.append(scaled_value)
            train_normalized_features.append(normalized_row)

        for sample in self.test_x:
            normalized_row = []
            for j in range(feature_count):
                column_data = [row[j] for row in self.data_x]
                min_val = min(column_data)
                max_val = max(column_data)
                if max_val == min_val:
                    scaled_value = 0
                else:
                    scaled_value = (sample[j] - min_val) / (max_val - min_val)
                normalized_row.append(scaled_value)
            test_normalized_features.append(normalized_row)

        # Convert lists to arrays for models
        self.train_x = np.array(train_normalized_features)
        self.test_x = np.array(test_normalized_features)

        # Make one-hot labels for MLP if needed
        unique_classes = list(set(self.data_y))
        class_count = len(unique_classes)

        if class_count > 2:
            # Create one-hot training labels
            self.train_y_onehot = []
            for label in self.train_y:
                onehot_row = [0] * class_count
                class_idx = unique_classes.index(label)
                onehot_row[class_idx] = 1
                self.train_y_onehot.append(onehot_row)

            # Create one-hot testing labels
            self.test_y_onehot = []
            for label in self.test_y:
                onehot_row = [0] * class_count
                class_idx = unique_classes.index(label)
                onehot_row[class_idx] = 1
                self.test_y_onehot.append(onehot_row)

            # Convert to arrays
            self.train_y_onehot = np.array(self.train_y_onehot)
            self.test_y_onehot = np.array(self.test_y_onehot)
        else:
            self.train_y_onehot = self.train_y
            self.test_y_onehot = self.test_y
        # <==== insert your code above here

    def run_comparison(self):
        """ Method to perform a fair comparison of three supervised machine learning algorithms.
        Should be extendable to include more algorithms later.

        For each of the algorithms KNearest Neighbour, DecisionTreeClassifier and MultiLayerPerceptron
        - Applies hyper-parameter tuning to find the best combination of relevant values for the algorithm
         -- creating and fitting model for each combination,
            then storing it in the relevant list in a dictionary called self.stored_models
            which has the algorithm names as the keys and lists of stored models as the values
         -- measuring the accuracy of each model on the test set
         -- keeping track of the best performing model for each algorithm, and its index in the relevant list so it can be retrieved.
        """
        # ====> insert your code below here
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neural_network import MLPClassifier

        # KNN: Try different k values
        k_options = [1, 3, 5, 7, 9]
        knn_index = 0
        for k in k_options:
            # Build and train KNN model
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            knn_classifier.fit(self.train_x, self.train_y)
            self.stored_models["KNN"].append(knn_classifier)

            # Compute accuracy
            correct_count = 0
            total_samples = len(self.test_y)
            predictions = knn_classifier.predict(self.test_x)
            for j in range(total_samples):
                if predictions[j] == self.test_y[j]:
                    correct_count += 1
            accuracy_percent = (correct_count / total_samples) * 100

            # Save best KNN model
            if accuracy_percent > self.best_accuracy["KNN"]:
                self.best_accuracy["KNN"] = accuracy_percent
                self.best_model_index["KNN"] = knn_index
            knn_index += 1

        # Decision Tree: Try different settings
        depth_options = [1, 3, 5]
        split_options = [2, 5, 10]
        leaf_options = [1, 5, 10]
        tree_index = 0
        for depth in depth_options:
            for split in split_options:
                for leaf in leaf_options:
                    # Build and train tree model
                    tree_classifier = DecisionTreeClassifier(
                        max_depth=depth, min_samples_split=split, min_samples_leaf=leaf, random_state=12345
                    )
                    tree_classifier.fit(self.train_x, self.train_y)
                    self.stored_models["DecisionTree"].append(tree_classifier)

                    # Compute accuracy
                    correct_count = 0
                    total_samples = len(self.test_y)
                    predictions = tree_classifier.predict(self.test_x)
                    for j in range(total_samples):
                        if predictions[j] == self.test_y[j]:
                            correct_count += 1
                    accuracy_percent = (correct_count / total_samples) * 100

                    # Save best tree model
                    if accuracy_percent > self.best_accuracy["DecisionTree"]:
                        self.best_accuracy["DecisionTree"] = accuracy_percent
                        self.best_model_index["DecisionTree"] = tree_index
                    tree_index += 1

        # MLP: Try different configurations
        first_layer_options = [2, 5, 10]
        second_layer_options = [0, 2, 5]
        activation_options = ["logistic", "relu"]
        mlp_index = 0
        for first in first_layer_options:
            for second in second_layer_options:
                for act in activation_options:
                    # Set layer sizes
                    if second == 0:
                        layers = (first,)
                    else:
                        layers = (first, second)

                    # Build and train MLP model
                    mlp_classifier = MLPClassifier(
                        hidden_layer_sizes=layers, activation=act, max_iter=1000, random_state=12345
                    )
                    mlp_classifier.fit(self.train_x, self.train_y_onehot)
                    self.stored_models["MLP"].append(mlp_classifier)

                    # Compute accuracy
                    correct_count = 0
                    total_samples = len(self.test_y)
                    predictions = mlp_classifier.predict(self.test_x)
                    num_classes = len(set(self.data_y))
                    if num_classes > 2:
                        # Handle one-hot predictions
                        for j in range(total_samples):
                            predicted_idx = 0
                            max_pred = predictions[j][0]
                            for k in range(1, num_classes):
                                if predictions[j][k] > max_pred:
                                    max_pred = predictions[j][k]
                                    predicted_idx = k
                            true_idx = 0
                            max_true = self.test_y_onehot[j][0]
                            for k in range(1, num_classes):
                                if self.test_y_onehot[j][k] > max_true:
                                    max_true = self.test_y_onehot[j][k]
                                    true_idx = k
                            if predicted_idx == true_idx:
                                correct_count += 1
                    else:
                        # Direct comparison for binary
                        for j in range(total_samples):
                            if predictions[j] == self.test_y[j]:
                                correct_count += 1
                    accuracy_percent = (correct_count / total_samples) * 100

                    # Save best MLP model
                    if accuracy_percent > self.best_accuracy["MLP"]:
                        self.best_accuracy["MLP"] = accuracy_percent
                        self.best_model_index["MLP"] = mlp_index
                    mlp_index += 1
        # <==== insert your code above here

    def report_best(self):
        """Method to analyse results.

        Returns
        -------
        accuracy: float
            the accuracy of the best performing model

        algorithm: str
            one of "KNN","DecisionTree" or "MLP"

        model: fitted model of relevant type
            the actual fitted model to be interrogated by marking code.
        """
        # ====> insert your code below here
        # Pick the best model overall
        max_accuracy = 0
        best_algo = ""
        best_classifier = None
        algo_list = ["KNN", "DecisionTree", "MLP"]
        for algo in algo_list:
            if self.best_accuracy[algo] > max_accuracy:
                max_accuracy = self.best_accuracy[algo]
                best_algo = algo
                best_classifier = self.stored_models[algo][self.best_model_index[algo]]
        return max_accuracy, best_algo, best_classifier
        # <==== insert your code above here
