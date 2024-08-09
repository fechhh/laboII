# Librerias necesarias
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import optuna


# Set current directory
#current_dir = os.getcwd()
#print("Current Directory:", current_dir)

#os.chdir("C:/Users/fgrijalba/Desktop/MCD/LaboII/laboII/PetFinder/workflows")

#--------------------------------------------------------------------
# Incorporo el dataset

def DT_incoporar_dataset():
    
    # Ruta Datasets
    train_path = '../petfinder_dataset/train/train.csv'
    test_path = '../petfinder_dataset/test/test.csv'
    
    # Cargo el dataset de entrenamiento y testeo
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Modifico el tipo de datos
    df_train['Vaccinated'] = df_train['Vaccinated'].astype('category')
    df_train['Dewormed'] = df_train['Dewormed'].astype('category')
    df_train['Sterilized'] = df_train['Sterilized'].astype('category')
    
    return df_train, df_test


#--------------------------------------------------------------------
# Catastrophe Analysis

def CA_catastrophe_analysis(df):
    
    # Cambio variables no especificadas (ejemplo: 0) a NA
    df['MaturitySize'] = df['MaturitySize'].replace(0, pd.NA)
    df['FurLength'] = df['FurLength'].replace(0, pd.NA)
    df['Vaccinated'] = df['Vaccinated'].replace(3, pd.NA)
    df['Dewormed'] = df['Dewormed'].replace(3, pd.NA)
    df['Sterilized'] = df['Sterilized'].replace(3, pd.NA)
    df['Health'] = df['Health'].replace(0, pd.NA)
    
    # Elimino variables no necesarias
    df = df.drop(["Name", "RescuerID", "Description"], axis=1)
    
    return df


#--------------------------------------------------------------------
# Feature Engineering manual

def FE_manual(df_train, df_test):
    
    # Misma raza
    df_train['SameBreed'] = np.where(df_train['Breed1'] == df_train['Breed2'], 1, 0)
    df_test['SameBreed'] = np.where(df_test['Breed1'] == df_test['Breed2'], 1, 0)

    # Description in english (VER!)
    #df['DescEnglish'] = np.where(df['Description'].str.contains('english', case=False), 1, 0)

    return df_train, df_test

#--------------------------------------------------------------------
# Feature Engineering random forest

def FErf_attributes_base(df_train, df_test):
    
    # Select only numerical columns, excluding the target column
    numerical_cols = df_train.select_dtypes(include=np.number).columns.drop("AdoptionSpeed")

    # Create X and y for training and testing sets
    X_train = df_train[numerical_cols]
    y_train = df_train["AdoptionSpeed"]
    
    X_test = df_test[numerical_cols]

    # Train RandomForest model
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    # Print feature importance
    feature_scores = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nFeature Importance:\n", feature_scores, "\n")
    
    # Predict the leaves of each tree in the forest
    train_leaves = rf.apply(X_train)
    test_leaves = rf.apply(X_test)
    
    # Convert leaves to DataFrame
    train_leaves_df = pd.DataFrame(train_leaves, columns=[f'leaf_{i}' for i in range(train_leaves.shape[1])])
    test_leaves_df = pd.DataFrame(test_leaves, columns=[f'leaf_{i}' for i in range(test_leaves.shape[1])])
    
    # Combine the original DataFrame with the leaves DataFrame
    df_train_combined = pd.concat([df_train.reset_index(drop=True), train_leaves_df], axis=1)
    df_test_combined = pd.concat([df_test.reset_index(drop=True), test_leaves_df], axis=1)
    
    return df_train_combined, df_test_combined


#--------------------------------------------------------------------

# Canaritos Asesinos


#--------------------------------------------------------------------

# Training Strategy


#--------------------------------------------------------------------

# Hyperparameter Tuning

def hyperparameter_tuning(df):
    # Select features and target variable
    X = df.drop(["AdoptionSpeed", "PetID"], axis=1)
    y = df["AdoptionSpeed"]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Create the LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)

    # Define the objective function for optimization
    def objective(trial):
        # Set the hyperparameters to be tuned
        params = {
            "objective": "multiclass",
            "num_class": 5,
            "metric": "multi_logloss",
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "verbose": -1
        }

        # Train the LightGBM model
        model = lgb.train(params, train_data, num_boost_round=100)

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred == y_test)

        return accuracy

    # Create the Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)   # VERRRRR

    # Get the best hyperparameters
    best_params = study.best_params
    print("\nBest Hyperparameters:", best_params)

    return best_params


#--------------------------------------------------------------------

# Final Model LightGBM

def final_model_lightgbm(df_train, df_test, best_params):
    
    # Check amount of features in each df
    print("Number of features in training set:", df_train.shape[1])
    print("Number of features in test set:", df_test.shape[1])
    
    # Check if the df have different features
    print("Features in training set but not in test set:", set(df_train.columns) - set(df_test.columns))
    
    
    # Separate the PetID column
    PetID = df_test['PetID']
    # Drop the PetID column
    df_test = df_test.drop("PetID", axis=1)
    
    # Select features and target variable
    X = df_train.drop(["AdoptionSpeed", "PetID"], axis=1)
    y = df_train["AdoptionSpeed"]

    # Create the LightGBM dataset
    train_data = lgb.Dataset(X, label=y)

    # Train the LightGBM model
    model = lgb.train(best_params, train_data, num_boost_round=100)
    
    # Make predictions and save them to a CSV file
    adopt_pred = model.predict(df_test)
    
    write_predictions_to_csv(model, df_test, PetID)

    return model


def write_predictions_to_csv(model, df_test, PetID, output_file="predictions.csv"):
    # Predict AdoptionSpeed using the final model
    y_pred = model.predict(df_test)

    # If y_pred is 1-dimensional, it's already class labels; otherwise, use np.argmax
    if y_pred.ndim == 1:
        y_pred_labels = y_pred
    else:
        y_pred_labels = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

    # Convert predictions to integers
    y_pred_labels = y_pred_labels.astype(int)

    # Create a DataFrame with PetID and predicted AdoptionSpeed
    predictions_df = pd.DataFrame({
        "PetID": PetID,
        "AdoptionSpeed": y_pred_labels
    })

    # Save the DataFrame to a CSV file
    predictions_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    
    

#--------------------------------------------------------------------

# Scoring




#--------------------------------------------------------------------
#--------------------------------------------------------------------
# Aca empieza el programa

def wf_adoption_speed():
    
    df_train, df_test = DT_incoporar_dataset()
    df_train = CA_catastrophe_analysis(df_train)
    df_test = CA_catastrophe_analysis(df_test)
    df_train, df_test = FE_manual(df_train, df_test)
    df_train, df_test = FErf_attributes_base(df_train, df_test)
    best_params = hyperparameter_tuning(df_train)
    final_model = final_model_lightgbm(df_train, df_test, best_params)
    
    print(df_test.head())



#--------------------------------------------------------------------
# Corro el programa

wf_adoption_speed()