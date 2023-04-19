import os
import sys

import pandas as pd
import numpy as np


def roi_ratio(df):
    # Get headers that contain "x" or "y" in their name
    headers = [col for col in df.columns if "x" in col or "y" in col]

    # Separate the data based on the second character of the header string
    r_headers = [h for h in headers if h[1] == "r"]
    l_headers = [h for h in headers if h[1] == "l"]

    # Calculate difference in real time for "r" dataset
    r_diff_x = np.max(df[r_headers[::2]], axis=1) - np.min(df[r_headers[::2]], axis=1)
    r_diff_y = np.max(df[r_headers[1::2]], axis=1) - np.min(df[r_headers[1::2]], axis=1)

    # Calculate difference in real time for "l" dataset
    l_diff_x = np.max(df[l_headers[::2]], axis=1) - np.min(df[l_headers[::2]], axis=1)
    l_diff_y = np.max(df[l_headers[1::2]], axis=1) - np.min(df[l_headers[1::2]], axis=1)

    # Calculate area for "r" dataset
    r_area = r_diff_x * r_diff_y

    # Calculate area for "l" dataset
    l_area = l_diff_x * l_diff_y


    # Return results as dictionary
    return r_diff_x, r_diff_y, r_area, l_diff_x, l_diff_y, l_area



def distance_nocche_to_zero(df, P1x, P1y, P2x, P2y, P3x, P3y, P4x, P4y, zero_x, zero_y):
    # Calculate the center of the 4 "nocche" points
    center_x = (df[P1x] + df[P2x] + df[P3x] + df[P4x]) / 4
    center_y = (df[P1y] + df[P2y] + df[P3y] + df[P4y]) / 4

    # Calculate the absolute distance to the zero point
    dist_x = np.abs(center_x - df[zero_x])
    dist_y = np.abs(center_y - df[zero_y])
    dist = np.sqrt(dist_x ** 2 + dist_y ** 2)

    # Calculate the projection on x and y axes
    proj_x = dist_x
    proj_y = dist_y

    # Output the results as numpy arrays
    return np.array(dist), np.array(proj_x), np.array(proj_y)

def add_handedness_to_dataset(input_df, output_df):
    # Get last two columns of input DataFrame
    handedness = input_df.iloc[:, -2:]

    # Add handedness to output DataFrame
    output_df = pd.concat([output_df, handedness], axis=1)

    return output_df

def save_dataframe_to_csv(df, file_name):
    # Create directory for saved files if it does not exist
    if not os.path.exists('features_data'):
        os.makedirs('features_data')

    # Save DataFrame to CSV file in features_data directory with headers
    file_path = os.path.join('features_data', file_name)
    df.to_csv(file_path, index=False)


def add_array_as_column(df, arr, col_name):
    # Convert NumPy array to Pandas Series
    arr_series = pd.Series(arr)

    # Add new column to DataFrame
    df[col_name] = arr_series

    return df

def delete_z_columns(df):
    # Get list of column headers starting with "z"
    z_cols = [col for col in df.columns if col.startswith('z')]

    # Delete columns starting with "z" from DataFrame
    df = df.drop(columns=z_cols)

    return df


def duplicate_first_5_cols(df):
    # Select first 5 columns of DataFrame
    first_5_cols = df.iloc[:, :5]

    # Create copy of first 5 columns
    first_5_cols_copy = first_5_cols.copy()

    return first_5_cols_copy



def extract_data(folder_path):
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return None

    # Get list of all CSV files in folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]
    if len(csv_files) == 0:
        print(f"No CSV files found in folder {folder_path}.")
        return None

    # Initialize list to store data from all CSV files
    data = []

    # Loop over all CSV files in folder
    for csv_file in csv_files:
        # Read CSV file into a Pandas DataFrame
        df = pd.read_csv(os.path.join(folder_path, csv_file))

        # Extract columns with headers that do not start with "z"
        df = delete_z_columns(df)

        df_features = duplicate_first_5_cols(df)
        df_features = add_handedness_to_dataset(df,df_features)


        #______________________FEATURES ENGINEERING_____________________#
        #inserisci qui le funzioni per il calcolo delle features
        #esempio features: moltiplicazione tra due colonne

        #DISTANZA NOCCHE BARI DA PUNTO ZERO
        distl,projxl,projyl = distance_nocche_to_zero(df,"xl5","yl5","xl9","yl9","xl13","yl13","xl17","yl17","xl0","yl0")
        distr,projxr,projyr = distance_nocche_to_zero(df,"xr5","yr5","xr9","yr9","xr13","yr13","xr17","yr17","xr0","yr0")
        df_features = add_array_as_column(df_features, distl, "dist_nocche_zero_l")
        df_features = add_array_as_column(df_features, projxl, "dist_nocche_zero_lx")
        df_features = add_array_as_column(df_features, projyl, "dist_nocche_zero_ly")
        df_features = add_array_as_column(df_features,distr,"dist_nocche_zero_r")
        df_features = add_array_as_column(df_features,projxr,"dist_nocche_zero_rx")
        df_features = add_array_as_column(df_features,projyr,"dist_nocche_zero_ry")

        r_ROI_x, r_ROI_y, r_ROI, l_ROI_x, l_ROI_y, l_ROI = roi_ratio(df)
        df_features = add_array_as_column(df_features, l_ROI_x, "l_ROI_x")
        df_features = add_array_as_column(df_features, l_ROI_y, "l_ROI_y")
        df_features = add_array_as_column(df_features, l_ROI, "l_ROI")
        df_features = add_array_as_column(df_features, r_ROI_x, "r_ROI_x")
        df_features = add_array_as_column(df_features, r_ROI_y, "r_ROI_y")
        df_features = add_array_as_column(df_features, r_ROI, "r_ROI")

        #________________SAVE_RESULTS____________________________________#
        save_dataframe_to_csv(df_features,"features_"+csv_file)

        print(df_features)

        #sblocca per processare tutti i dati

        sys.exit()

    # Concatenate data from all CSV files into a single NumPy array

    return data


folder_path = "mediapipe_raw_data"
data_list = extract_data(folder_path)

# Concatenate data from all files into a single NumPy array


