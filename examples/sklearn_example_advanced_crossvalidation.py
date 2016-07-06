from __future__ import print_function, division
import ac_pysmac
import sklearn.ensemble
import sklearn.datasets
import sklearn.cross_validation

# We use the same data as the earlier sklearn example.
X, Y = sklearn.datasets.make_classification(1000, 20,
                                            random_state=2)  # seed yields a mediocre initial accuracy on my machine

# But this time, we do not split it into train and test data set, but we will use
# k-fold cross validation insteat to estimate the accuracy better. Henre we shall
# use k=10 for demonstration purposes. To make thins more convinient later on,
# let's convert the KFold iterator into a list, so we can use indexing.
kfold = [(test, train) for (test, train) in sklearn.cross_validation.KFold(X.shape[0], 10)]


# We have to make a slight modification to the function fitting the random forest.
# it now has to take an additional argument instance. (Note: SMAC grew historically
# in the context of algorithm configuration, where the performance across multiple
# instances is optimized. The naming convention is a tribute to that heritage.)
# This argument will be a integer between 0 and num_instances (defined below).
# Note that this increases the computational effort as SMAC now estimates the
# quality of a parameter setting for multiple instances.
def random_forest(n_estimators, criterion, max_features, max_depth, instance):
    # Use the requested fold
    train, test = kfold[instance]
    X_train, Y_train, X_test, Y_test = X[train], Y[train], X[test], Y[test]

    predictor = sklearn.ensemble.RandomForestClassifier(n_estimators, criterion, max_features, max_depth)
    predictor.fit(X_train, Y_train)

    return -predictor.score(X_test, Y_test)


# We haven't changed anything here.
parameter_definition = dict( \
    max_depth=("integer", [1, 10], 4),
    max_features=("integer", [1, 20], 10),
    n_estimators=("integer", [1, 100], 10, 'log'),
    criterion=("categorical", ['gini', 'entropy'], 'entropy'),
)

# Same creation of the SMAC_optimizer object
opt = ac_pysmac.SMAC_optimizer(working_directory='/tmp/ac_pysmac_test/',  # the folder where SMAC generates output
                               persistent_files=False,
                               # whether the output will persist beyond the python object's lifetime
                               debug=False  # if something goes wrong, enable this for diagnostic output
                               )

# first we try the sklearn default, so we can see if SMAC can improve the performance
accuracy = 0.

for train, test in kfold:
    X_train, Y_train, X_test, Y_test = X[train], Y[train], X[test], Y[test]
    predictor = sklearn.ensemble.RandomForestClassifier()
    predictor.fit(X_train, Y_train)
    accuracy += predictor.score(X_test, Y_test)

print('The default accuracy is %f' % (accuracy / len(kfold)))

# The minimize method also has optional arguments
value, parameters = opt.minimize(random_forest,
                                 500, parameter_definition,
                                 num_runs=2,  # number of independent SMAC runs
                                 seed=0,  # the random seed used. can be an int or a list of ints of length num_runs
                                 num_procs=2,
                                 # ac_pysmac can harness multicore architecture. Specify the number of processes to use here.
                                 num_train_instances=len(kfold)
                                 # This tells SMAC how many different instances there are.
                                 )

print('The highest accuracy found: %f' % (-value))
print('Parameter setting %s' % parameters)
