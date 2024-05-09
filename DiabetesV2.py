# Probabilistic Graphical Model
# Diabetes Diagnosis 
# 2427724

# --------------Imports--------------------
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator, HillClimbSearch, BDeuScore, BicScore, K2Score
from pgmpy.estimators import ScoreCache
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
# --------------Imports--------------------


def plot_structure(DAG):
    G = nx.DiGraph()
    G.add_nodes_from(DAG.nodes)
    G.add_edges_from(DAG.edges)
    pos = nx.shell_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=500, arrowsize=20)
    plt.show()

def preprocess_data(data):
    
    # Handle missing data point
    # Learn the structure of the Bayesian network
    model = BayesianModel()
    model.add_nodes_from(data.columns)
    hc = HillClimbSearch(data)
    best_model = hc.estimate(scoring_method=BicScore(data))
    model.add_edges_from(best_model.edges())

    # Fit the Bayesian network to the data
    model.fit(data, estimator=BayesianEstimator, state_names=data.columns)

    # Fill in missing values using Variable Elimination
    inference = VariableElimination(model)
    filled_data = data.copy()

    for index, row in filled_data.iterrows():
        missing_features = row[row.isna()].index
        if not missing_features.empty:
            evidence = row.drop(missing_features).to_dict()
            for feature in missing_features:
                query_result = inference.query([feature], evidence=evidence)
                filled_data.at[index, feature] = query_result.values[0]
                                                                    

    # Bin numerical columns
    bins = 4
    data['BMI'] = pd.cut(data['BMI'], bins=bins, labels=False)
    data['MentHlth'] = pd.cut(data['MentHlth'], bins=bins, labels=False)
    data['PhysHlth'] = pd.cut(data['PhysHlth'], bins=bins, labels=False)

    # Perform feature selection
    corr_matrix = data.corr()
    target_corr = corr_matrix['Diabetes_binary']
    threshold = 0.14  # threshold (correlation between given feature and target)
    selected_features = target_corr[abs(target_corr) > threshold].index.tolist()
    selected_features.remove('Diabetes_binary')
    data = data[selected_features + ['Diabetes_binary']]

    # Oversample the minority class
    # Separate features and target variable
    X = data.drop('Diabetes_binary', axis=1)
    y = data['Diabetes_binary']

    # Perform oversampling
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Combine the oversampled features and target variable back into a DataFrame
    data_resampled = pd.concat([X_resampled, y_resampled], axis=1)

    # Min Max Scaling
    scaler = MinMaxScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)
    data_resampled = pd.concat([pd.DataFrame(X_resampled_scaled, columns=X.columns), y_resampled], axis=1)

    return data_resampled

def learn_structure(training_data, validation_data, num_random_restarts=3):
    best_model = None
    best_loglik = -float('inf')

    for _ in range(num_random_restarts):
        best_score = -float('inf')
        best_model_restart = None

        for scoring_method in [K2Score, BDeuScore, BicScore]:
            hc = HillClimbSearch(training_data)
            estimator = scoring_method(training_data)

            add_node = random.random()
            which_node = random.random()
            start_dag = generate_random_dag(training_data.columns, add_node, which_node)

            best_structure = hc.estimate(scoring_method=estimator, start_dag=start_dag)

            current_model = BayesianNetwork(best_structure.edges())
            current_model.fit(training_data, estimator=BayesianEstimator)

            loglik = log_likelihood_score(current_model, validation_data)

            if loglik > best_score:
                best_score = loglik
                best_model_restart = current_model

        if best_score > best_loglik:
            best_loglik = best_score
            best_model = best_model_restart

    return best_model, best_loglik


# -------- CHANGE THIS CODE ------------------ 
def generate_random_dag(nodes, add_node, which_node):
    dag = DAG()
    dag.add_nodes_from(nodes)

    # creates a random direct acyclic graph
    node_count = len(nodes)
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if random.random() < add_node:
                if random.random() < which_node:
                    edge = (nodes[i], nodes[j])
                else:
                    edge = (nodes[j], nodes[i])

                # Check if adding the edge creates a cycle
                if not nx.has_path(dag, edge[1], edge[0]):
                    dag.add_edge(*edge)

    plot_structure(dag)
    return dag

# def generate_random_dag(num_nodes, num_edges, variables):
#     if num_nodes < 1:
#         raise ValueError("Number of nodes must be at least 1.")
#     if num_edges < 0:
#         raise ValueError("Number of edges cannot be negative.")

#     if num_edges > num_nodes * (num_nodes - 1) / 2:
#         raise ValueError("Number of edges exceeds the maximum possible for a DAG with {} nodes.".format(num_nodes))

#     G = nx.DiGraph()
#     G.add_nodes_from(variables)

#     edge_count = 0
#     while edge_count < num_edges:
#         u = random.choice(variables)
#         v = random.choice(variables)
#         if u != v and not G.has_edge(u, v) and not nx.has_path(G, v, u):
#             G.add_edge(u, v)
#             edge_count += 1

#     plot_structure(G)
#     return G

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
    # fetch dataset 
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

    # This gives us a 70:20:10 split
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=8)
    # Split the training data into training and validation sets
    train_data, validation_data = train_test_split(train_data, test_size=0.3, random_state=8)

    # Learn the structure of the Bayesian network using the best scoring function
    best_model, best_loglik = learn_structure(train_data, validation_data)

    # Choose the best model based on the log-likelihood score
    # best_model, best_loglik = max([(best_model_bdeu, loglik_bdeu), (best_model_bic, loglik_bic), (best_model_k2, loglik_k2)], key=lambda x: x[1])
    # print(f"Best scoring function: {type(best_model).__name__}")

    # Make predictions on the test set
    test_data_without_outcome = test_data.drop('Diabetes_binary', axis=1)
    predicted_outcomes = []
    for _, row in test_data_without_outcome.iterrows():
        predicted_outcome = predict_diabetes(best_model, row.to_dict())
        predicted_outcomes.append(predicted_outcome)

    # Calculate evaluation metrics
    actual_outcomes = test_data['Diabetes_binary'].tolist()
    accuracy = accuracy_score(actual_outcomes, predicted_outcomes)
    precision = precision_score(actual_outcomes, predicted_outcomes)
    Sensitivity = recall_score(actual_outcomes, predicted_outcomes)
    f1 = f1_score(actual_outcomes, predicted_outcomes)

    # Print accuracies
    print(f"Accuracy: {round(accuracy*100,0)}%")
    print(f"Precision: {round(precision*100,0)}%")
    print(f"Sensitivity: {round(Sensitivity*100,0)}%")
    print(f"F1-score: {round(f1*100,0)}%")

    # Plot the structure of the best model
    plot_structure(best_model)

    # Plot the confusion matrix
    cm = confusion_matrix(actual_outcomes, predicted_outcomes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Diagnosis')
    plt.ylabel('Actual Diagnosis')
    plt.title('Confusion Matrix')
    plt.show()



if __name__ == '__main__':
    main()