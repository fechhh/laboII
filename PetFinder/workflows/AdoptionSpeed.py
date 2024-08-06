# Librerias necesarias
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split


# Set current directory
#current_dir = os.getcwd()
#print("Current Directory:", current_dir)

#os.chdir("C:/Users/fgrijalba/Desktop/MCD/LaboII/laboII/PetFinder/workflows")

#--------------------------------------------------------------------
# Incorporo el dataset

def DT_incoporar_dataset():
    
    # Ruta Dataset
    file_path = '../petfinder_dataset/train/train.csv'
    
    df = pd.read_csv(file_path)
    
    return df


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
    
    return df


#--------------------------------------------------------------------
# Feature Engineering manual

def FE_manual(df):
    
    # Misma raza
    df['SameBreed'] = np.where(df['Breed1'] == df['Breed2'], 1, 0)

    # Description in english (VER!)
    df['DescEnglish'] = np.where(df['Description'].str.contains('english', case=False), 1, 0)

    return df

#--------------------------------------------------------------------
# Feature Engineering random forest

def FErf_attributes_base(df):
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=np.number).columns

    # Create X and y using numerical columns
    X = df[numerical_cols].drop(["AdoptionSpeed"], axis=1)
    y = df["AdoptionSpeed"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predict the leaves of each tree in the forest
    train_leaves = rf.apply(X_train)
    test_leaves = rf.apply(X_test)
    
    # Combine train and test leaves
    leaves = np.concatenate((train_leaves, test_leaves), axis=0)
    leaves_df = pd.DataFrame(leaves, columns=[f'leaf_{i}' for i in range(leaves.shape[1])])
    
    # Combine the original dataframe with the leaves dataframe
    df_combined = pd.concat([df.reset_index(drop=True), leaves_df], axis=1)
    
    return df_combined


#--------------------------------------------------------------------

# Canaritos Asesinos


#--------------------------------------------------------------------

# Training Strategy


#--------------------------------------------------------------------

# Hyperparameter Tuning


#--------------------------------------------------------------------

# Final Model LightGBM


#--------------------------------------------------------------------

# Scoring




#--------------------------------------------------------------------
#--------------------------------------------------------------------
# Aca empieza el programa

def wf_adoption_speed():
    
    df = DT_incoporar_dataset()
    df = CA_catastrophe_analysis(df)
    df = FE_manual(df)
    df = FErf_attributes_base(df)
    
    print(df.head())

wf_adoption_speed()