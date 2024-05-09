# Probabilistic Graphical Model
# Diabetes Diagnosis 
# 2427724

# --------------Imports--------------------
import pandas as pd
from pgmpy.estimators import ScoreCache
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator, HillClimbSearch, BDeuScore, BicScore, K2Score
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import BayesianNetwork
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from pgmpy.metrics import log_likelihood_score
import networkx as nx
import random
from pgmpy.base import DAG
# -----------------------------------------------------

def plot_structure(graph):
    # Simple structure plotter
    # use nx package to plot
    G = nx.DiGraph()

    # add the nodes
    G.add_nodes_from(graph.nodes)

    # add the edges
    G.add_edges_from(graph.edges)

    # use shell layout
    pos = nx.shell_layout(G)

    # specify sizes
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=480, arrowsize=25)

    # Show the plot
    plt.show()

def preprocess_data(data):
    # Handle missing data
    # Create a Bayesian network model
    model = BayesianModel()
    model.add_nodes_from(data.columns)
    
    # Perform structure learning using hill climb search
    search = HillClimbSearch(data)
    learned_model = search.estimate(scoring_method=BicScore(data))
    model.add_edges_from(learned_model.edges())
    
    # Fit the Bayesian network to the data
    model.fit(data, estimator=BayesianEstimator, state_names=data.columns)
    
    # Handle missing values using Variable Elimination
    inference = VariableElimination(model)
    processed_data = data.copy()
    
    # Iterate over each row and fill in missing values
    for index, row in processed_data.iterrows():
        missing_features = row[row.isna()].index
        if not missing_features.empty:
            evidence = row.drop(missing_features).to_dict()
            for feature in missing_features:
                query_result = inference.query([feature], evidence=evidence)
                processed_data.at[index, feature] = query_result.values[0]
                                                                    
    # Bin numerical columns
    bins = 4
    data['BMI'] = pd.cut(data['BMI'], bins=bins, labels=False)
    data['MentHlth'] = pd.cut(data['MentHlth'], bins=bins, labels=False)
    data['PhysHlth'] = pd.cut(data['PhysHlth'], bins=bins, labels=False)

    # Perform feature selection
    corr_matrix = data.corr()

    # check correlation between features and target
    target_correlation = corr_matrix['Diabetes_binary']
    threshold = 0.1  # threshold (correlation between given feature and target)
    correlated_features = target_correlation[abs(target_correlation) > threshold].index.tolist()
    correlated_features.remove('Diabetes_binary')
    data = data[correlated_features + ['Diabetes_binary']]

    # Oversample the minority class
    # Separate features and target variable
    X = data.drop('Diabetes_binary', axis=1)
    y = data['Diabetes_binary']

    # Perform oversampling
    oversampler = RandomOverSampler(random_state=36) 
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Combine the oversampled features and target variable
    data_resampled = pd.concat([X_resampled, y_resampled], axis=1)

    # Min Max Scaling
    scaler = MinMaxScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)
    data_resampled = pd.concat([pd.DataFrame(X_resampled_scaled, columns=X.columns), y_resampled], axis=1)

    return data_resampled

def learn_structure(training_data, validation_data, num_random_restarts=3):
    # Initialize the best model and best log likelihood
    best_model = None
    best_loglikelihood = -float('inf')

    # Perform multiple random restarts
    for i in range(num_random_restarts):

        best_score = -float('inf')
        best_model_restart = None

        # Iterate over different scoring methods
        for scoring_method in [K2Score, BDeuScore, BicScore]:

            hc = HillClimbSearch(training_data)

            # Use the current scoring method
            estimator = scoring_method(training_data)

            # Randomly select whether to add a node or an edge
            add_node = random.random()
            pick_node = random.random()

            # Generate a random DAG with the same variables as the dataset
            generated_dag = create_random_dag(training_data.columns, add_node, pick_node)

            optimal_structure = hc.estimate(scoring_method=estimator, start_dag=generated_dag)

            current_model = BayesianNetwork(optimal_structure.edges())

            current_model.fit(training_data, estimator=BayesianEstimator)

            # Calculate the log-likelihood score on the validation data
            loglikelihood = log_likelihood_score(current_model, validation_data)

            # Update the best score and model if the current model is better
            if loglikelihood > best_score:
                best_score = loglikelihood
                best_model_restart = current_model

        # Update the overall best model if the best model of this restart is better
        if best_score > best_loglikelihood:
            best_loglikelihood = best_score
            best_model = best_model_restart

    return best_model, best_loglikelihood

def create_random_dag(node_list, edge_probability, reverse_edge_probability):
    # Create an empty directed acyclic graph
    graph = DAG()
    graph.add_nodes_from(node_list)
    
    num_nodes = len(node_list)
    
    # Iterate over each pair of nodes
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            # Check if an edge should be added based on the edge probability
            if random.random() < edge_probability:
                # Determine the direction of the edge based on the reverse edge probability
                if random.random() < reverse_edge_probability:
                    edge = (node_list[i], node_list[j])
                else:
                    edge = (node_list[j], node_list[i])
                
                # Add the edge if it doesn't create a cycle
                if not nx.has_path(graph, edge[1], edge[0]):
                    graph.add_edge(*edge)
    
    return graph

def fit_model(model, data):
    # Fit the model to the bayesian estimator
    model.fit(data, estimator=BayesianEstimator)
    return model

def predict_diabetes(model, data_point):
    # Perform variable elimination for inference
    inference = VariableElimination(model)

    # Query the model
    query = inference.query(variables=['Diabetes_binary'], evidence=data_point)

    # Get the predicted outcome
    predicted_outcome = query.values.argmax()

    return predicted_outcome

def fetch_data():
    # fetch diabetes dataset 
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
    
    # data (as pandas dataframes) 
    x = cdc_diabetes_health_indicators.data.features 
    y = cdc_diabetes_health_indicators.data.targets 

    # Put the targets and features in 1 data table
    data = pd.concat([x, y], axis=1)
    return data

def main():
    # Fetch the data
    data = fetch_data()

    # Preprocess the data
    data = preprocess_data(data)

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=8)
    # Split the training data into training and validation sets

    train_data, validation_data = train_test_split(train_data, test_size=0.3, random_state=8)

    # Learn the structure of the Bayesian network using the best scoring function
    best_model, best_loglik = learn_structure(train_data, validation_data)

    # Make predictions on the test set
    test_data_without_outcome = test_data.drop('Diabetes_binary', axis=1)
    predicted_outcomes = []
    for i, row in test_data_without_outcome.iterrows():
        predicted_outcome = predict_diabetes(best_model, row.to_dict())
        predicted_outcomes.append(predicted_outcome)

    # Calculate evaluation metrics
    actual_outcomes = test_data['Diabetes_binary'].tolist()

    # Calculate accuracy
    accuracy = accuracy_score(actual_outcomes, predicted_outcomes)

    # Calculate precision
    precision = precision_score(actual_outcomes, predicted_outcomes)

    # Calculate sensitivity
    Sensitivity = recall_score(actual_outcomes, predicted_outcomes)

    # Calculate f1 score based on above
    f1 = f1_score(actual_outcomes, predicted_outcomes)

    # Print accuracies
    print(f"Accuracy: {round(accuracy*100)}%")
    print(f"Precision: {round(precision*100)}%")
    print(f"Sensitivity: {round(Sensitivity*100)}%")
    print(f"F1-score: {round(f1*100)}%")

    # Plot the structure of the best model
    plot_structure(best_model)

    # Plot the confusion matrix
    cm = confusion_matrix(actual_outcomes, predicted_outcomes)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Diagnosis')
    plt.ylabel('Actual Diagnosis')
    plt.show()

if __name__ == '__main__':
    main()