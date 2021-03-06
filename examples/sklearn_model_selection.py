from __future__ import print_function, division
import ac_pysmac
import sklearn.ensemble
import sklearn.neighbors
import sklearn.datasets
import sklearn.cross_validation

# We use a random classification data set generated by sklearn
# As commonly done, we use a train-test split to avoid overfitting.
X, Y = sklearn.datasets.make_classification(1000, 20)
X_train, X_test, Y_train, Y_test = \
    sklearn.cross_validation.train_test_split(X, Y, test_size=0.33, random_state=1)


# Here, SMAC can choose between to different models at each evaluation. To
# make the search more efficient, it is important to tell SMAC that some
# parameters are associated with certain classifiers
def choose_classifier(classifier,  # which classifier to use
                      # parameters for the tree based classifiers
                      trees_n_estimators=None, trees_criterion=None,
                      trees_max_features=None, trees_max_depth=None,
                      # the ones for k-nearest-neighbors
                      knn_n_neighbors=None, knn_weights=None):
    # note that possibly inactive variables have to be optional
    # as ac_pysmac does not assign a value for inactive variables
    # during the minimization phase
    if classifier == 'random_forest':
        predictor = sklearn.ensemble.RandomForestClassifier(
            trees_n_estimators, trees_criterion,
            trees_max_features, trees_max_depth)
    elif classifier == 'extra_trees':
        predictor = sklearn.ensemble.ExtraTreesClassifier(
            trees_n_estimators, trees_criterion,
            trees_max_features, trees_max_depth)
    elif classifier == 'k_nearest_neighbors':
        predictor = sklearn.neighbors.KNeighborsClassifier(
            knn_n_neighbors, knn_weights)

    predictor.fit(X_train, Y_train)
    return -predictor.score(X_test, Y_test)


# defining all the parameters with respective defaults.
parameter_definition = dict( \
    trees_max_depth=("integer", [1, 10], 4),
    trees_max_features=("integer", [1, 20], 10),
    trees_n_estimators=("integer", [1, 100], 10, 'log'),
    trees_criterion=("categorical", ['gini', 'entropy'], 'entropy'),
    knn_n_neighbors=("integer", [1, 100], 10, 'log'),
    knn_weights=("categorical", ['uniform', 'distance'], 'uniform'),
    classifier=("ordinal", ['random_forest', 'extra_trees', 'k_nearest_neighbors'], 'random_forest'),
    # Usually you would make this a categorical, but to showcase all
    # conditional clauses, let's pretend it's an ordinal parameter,
    # so we can use > and <.
)

# here we define the dependencies between the parameters. the notation is
#   <child> | <parent> in { <parent value>, ... }
# and means that the child parameter is only active if the parent parameter
# takes one of the value in the listed set. The notation follows the SMAC
# manual one to one. Note there is no checking for correctness beyond
# what SMAC does. I.e., when you have a typo in here, you don't get any 
# meaningful output, unless you set  debug = True below!
conditionals = ['trees_max_depth    | classifier in {random_forest, extra_trees}',
                'trees_max_features | classifier in {random_forest} || classifier == extra_trees',
                'trees_n_estimators | classifier != k_nearest_neighbors',
                'trees_criterion    | classifier < k_nearest_neighbors',
                'knn_n_neighbors    | classifier > extra_trees',
                'knn_weights        | classifier == k_nearest_neighbors && classifier != extra_trees && classifier != random_forest'
                ]

# creation of the SMAC_optimizer object. Notice the optional debug flag
opt = ac_pysmac.SMAC_optimizer(debug=0,
                               working_directory='/tmp/ac_pysmac_test/', persistent_files=True, )

# first we try the sklearn default, so we can see if SMAC can improve the performance

predictor = sklearn.ensemble.RandomForestClassifier()
predictor.fit(X_train, Y_train)
print('The default accuracy of the random forest is %f' % predictor.score(X_test, Y_test))

predictor = sklearn.ensemble.ExtraTreesClassifier()
predictor.fit(X_train, Y_train)
print('The default accuracy of the extremely randomized trees is %f' % predictor.score(X_test, Y_test))

predictor = sklearn.neighbors.KNeighborsClassifier()
predictor.fit(X_train, Y_train)
print('The default accuracy of k-nearest-neighbors is %f' % predictor.score(X_test, Y_test))

# The minimize method also has optional arguments (more on that in the section on advanced configuration).
value, parameters = opt.minimize(choose_classifier,
                                 500, parameter_definition,
                                 conditional_clauses=conditionals)

print('The highest accuracy found: %f' % (-value))
print('Parameter setting %s' % parameters)
