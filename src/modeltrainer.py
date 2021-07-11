import pandas as pd
import os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils import save_dict

class ModelTrainer():
    """
    Used for grid search training of algorithms, returning the best model
    """
    
    def __init__(self, estimator, dataset=None, filename=None, vectorizer=None):
        self.valid_datasets = ['yelp', 'clickbait', 'subjectivity_objectivity']

        # Set model
        self.estimator = estimator

        # Initialize vectorizer
        if vectorizer is None:
            self.vectorizer = CountVectorizer(analyzer='char', 
                                            tokenizer=word_tokenize, 
                                            stop_words=stopwords.words('english'))
        else:
            self.vectorizer = vectorizer

        # Load dataset
        if dataset or filename:
            self.load_dataset(dataset=dataset, filename=filename)
        
                    
    def load_dataset(self, dataset=None, filename=None):
        """
        Loads either a pre-packaged dataset or data set from file. 
        If both specified, dataset takes priority.

        Params:
        - dataset: str, optional (default=None)
            Name of pre-packaged dataset to load.
            If None: loads filename
        - filename: str, optional (default=None)
            Filename of user-specified dataset to load.
            If None: loads dataset
        """
        
        # Make sure there is a dataset to load
        try:
            assert dataset or filename
        except AssertionError:
            raise ValueError('Either dataset or file to load dataset from must be specified')
            
        # Ensure dataset is valid
        if dataset in self.valid_datasets:
            dataset_loader = getattr(self, f'_load_{dataset}')
            self.data = dataset_loader()
            self.data_name = dataset
        elif os.path.exists(filename):
            self.data = pd.read_csv(filename)

            # Takes the filename from the filename/absolute filepath, and then removes the extension
            self.data_name = filename.split(os.sep)[-1].split('.')[0]
        else:
            raise ValueError(f'Both dataset={dataset} and filename={filename} are invalid data options')
            
    def _load_yelp(self):
        # Load yelp data sets
        yelp_test_df = pd.read_csv('../data/yelp_review_polarity_csv/test.csv', names=['label', 'data']) 
        yelp_train_df = pd.read_csv('../data/yelp_review_polarity_csv/train.csv', names=['label', 'data']) 

        # Since yelp data set is already split into test and train, recombine
        yelp_df = pd.concat([yelp_test_df, yelp_train_df])

        # Data set is too large to work with in memory, so cut to workable size
        yelp_df = yelp_df.sample(n=32000,replace=False,axis='index')

        # Change 1, 2 label to 0, 1 for uniformity with other data sets
        # Data set has 1 for negative and 2 for positive, so we switch 0 to negative and 1 to positive
        yelp_df['label'] = yelp_df['label'].apply(lambda label: 0 if label == 1 else 1)

        # Transform data into vectorized format
        yelp_df['data'] = self.vectorizer.fit_transform(yelp_df['data']).toarray()

        # Return data as np array
        return yelp_df.values

    def _load_clickbait(self):
        # Load data sets
        clickbait_df = pd.read_csv('../data/clickbait/clickbait_data', sep='\n', names=['data'])
        nonclickbait_df = pd.read_csv('../data/clickbait/non_clickbait_data', sep='\n', names=['data'])

        # Add labels (clickbait is 0, non-clickbait is 1)
        nonclickbait_df['label'] = 0
        clickbait_df['label'] = 1

        # Combine data sets and rearrange columns for uniformity
        clickbait_df = pd.concat([clickbait_df, nonclickbait_df])
        clickbait_df = clickbait_df.reindex(columns=['label', 'data'])

        # Transform data into vectorized format
        clickbait_df['data'] = self.vectorizer.fit_transform(clickbait_df['data']).toarray()

        # Return data as np array
        return clickbait_df.values
        
    def _load_subjectivity_objectivity(self):
        # Load data sets
        subjectivity_df = pd.read_csv('../data/subjectobject/subjectivity.txt', sep='\n', encoding='latin-1', names=['data'])
        objectivity_df = pd.read_csv('../data/subjectobject/objectivity.txt', sep='\n', encoding='latin-1', names=['data'])

        # Add labels (subjective is 0, objective is 1)
        subjectivity_df['label'] = 0
        objectivity_df['label'] = 1

        # Combine data sets and rearrange columns for uniformity
        sub_ob_df = pd.concat([subjectivity_df, objectivity_df])
        sub_ob_df = sub_ob_df.reindex(columns=['label', 'data'])

        # Transform data into vectorized format
        sub_ob_df['data'] = self.vectorizer.fit_transform(sub_ob_df['data']).toarray()

        # Return data as np array
        return sub_ob_df.values

    def get_metric_dict(self):
        """
        Returns the metric dict for the most recent model run
        """

        try:
            return self.metric_dict
        except:
            raise ValueError('No runs to return metrics for')

    def get_best_model(self, param_grid, scoring='accuracy', n_jobs=1, verbose=1, save=False):
        """
        Takes data and model information and returns a dictionary of metrics on the best estimator for each data 
        set via grid search
        
        Params:
        - param_grid: (dict or list of dicts) 
            values to perform grid search over
        - scoring: str, optional (default='accuracy')
            specifies how to rank each estimator
        - n_jobs: int, optional (default='1')
            number of cores to run training on. -1 includes all cores
        - verbose: int, optional (default='1')
            specifies if output messages should be provided
        - save: bool, optional (default=False)
            flag for saving metric dict
        Returns:
        - clf: gridsearch object with best performance
        """
        
        # Make sure proper data was passed in
        try:
            assert type(param_grid) in [list, set, tuple, dict]
        except AssertionError:
            raise ValueError('Unexpected data type passed in for param_grid')
        try:
            if type(param_grid) is not dict:
                assert type(param_grid[0]) == dict
        except AssertionError:
            raise ValueError('Unexpected data type passed in for param_grid')
        
        self.metric_dict = dict()
        clf = GridSearchCV(estimator=self.estimator, 
                        param_grid=param_grid, 
                        cv=5, n_jobs=n_jobs, 
                        verbose=verbose, 
                        scoring=scoring)
        
        
        X, y = self.data[:, 1:], self.data[:, :1] #Treats first column as label
        for i in range(3): # Completes 3 trials
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000, shuffle=True)

            clf.fit(X_train, y_train.ravel()) # Fit training data to model

            # Gather training set metrics
            y_train_pred = clf.predict(X_train)
            acc_train = accuracy_score(y_train, y_train_pred)
            precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(y_train, y_train_pred)

            # Gather testing set metrics
            y_test_pred = clf.predict(X_test) # Predict test values using best parameters from classifier
            acc_test = accuracy_score(y_test, y_test_pred) # Get accuracy for predictions
            precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, y_test_pred)

            # Save metrics to dict for further analysis
            self.metric_dict[(self.data_name, i)] = {'acc_test': acc_test, 
                                    'acc_train': acc_train, 
                                    'precision_test': precision_test, 
                                    'precision_train': precision_train, 
                                    'recall_test': recall_test, 
                                    'recall_train': recall_train,
                                    'f1_test': f1_test, 
                                    'f1_train': f1_train, 
                                    'model': clf, 
                                    'cv_results': clf.cv_results_} # Add metrics to dict for analysis
            if save:
                # Save checkpoint results in case of hardware failure
                loc_str = self.estimator.__class__.__name__ # this just gets clf type (eg SVC, LogisticRegression, etc)
                
                # Checks if the output path already exists, and makes it if not
                save_dir = os.path.join('..', 'checkpoints', f'{loc_str}')
                if not os.path.isdir(save_dir):
                    print(f'Creating {loc_str} directory now')
                    os.mkdir(os.path.join('..', 'checkpoints', loc_str))
                    save_path = os.path.join(save_dir, f'{loc_str}_{self.data_name}_{i}.pkl')
                    save_dict(self.metric_dict, save_path)
        
        return clf