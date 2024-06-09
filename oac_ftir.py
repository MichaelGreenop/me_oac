import streamlit as st
import pandas as pd
import pickle
import numpy as np
from BaselineRemoval import BaselineRemoval
from scipy.signal import savgol_filter


# Preprocessing 

def BL(input_array):
    polynomial_degree=2
    
    baseObj=BaselineRemoval(input_array)
    Modpoly_output=baseObj.ModPoly(polynomial_degree)
    
    return Modpoly_output

def snv(input_data):
  
    # Define a new array and populate it with the corrected data  
    mean = np.mean(input_data)
    std_dev = np.std(input_data)
    snv_spec = (input_data - mean) / std_dev
 
    return snv_spec

def SG(input_data):
    sg_spec = savgol_filter(input_data, 41, polyorder = 3, deriv=2)
    return sg_spec




# Load the pre-trained model from a pickle file
@st.cache_resource
def load_model_one():
    with open('June_pca_m1_v0.pkl', 'rb') as file:
        model_one = pickle.load(file)
    return model_one

def load_model_two():
    with open('June_pca_m2_v0.pkl', 'rb') as file:
        model_two = pickle.load(file)
    return model_two

# Function to make predictions
def transform_1(model_one, data_1):
    values_1 = model_one.transform(data_1)
    return values_1

def transform_1(model_two, data_2):
    values_2 = model_two.transform(data_2)
    return values_2

# Streamlit app
def main():
    st.title("OAC Prediction App")
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")


    

    
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file, header=None)
        w = df[0]
        spec = df[1]

        waves = w.iloc[1:2076].values.astype(float).astype(int)
        data = spec.iloc[1:2076].values

        bl = BL(data)
        SNV = snv(bl)
        sg = SG(SNV).reshape((2075,1))

        

        pp_data = pd.DataFrame(sg).set_index(waves)

        # Model 1 test features
        test_spec_975 = pp_data.loc[975:981].mean(axis=0)
        test_spec_1026 = pp_data.loc[1026:1028].mean(axis=0)
        test_spec_1586 = pp_data.loc[1586:1588].mean(axis=0)

        # Model 2 test features
        test_spec_979 = pp_data.loc[979:981].mean(axis=0)
        test_spec_1025 = pp_data.loc[1025:1028].mean(axis=0)
        

        DF = pd.DataFrame([test_spec_975, test_spec_1026, test_spec_1586, test_spec_979, test_spec_1025])


        
        
        data_1 = np.array(DF).reshape((5,1))[:3].T
        data_2 = np.array(DF).reshape((5,1))[3:].T

        
                
        # Load the model
        model_1 = load_model_one()
        
        # Make predictions
        value_1 = transform_1(model_1, data_1)
                
        # Display predictions
        st.write("Predictions:")
        if value_1 > 0.00006:
            st.write('Not OAC')
        else:
            model_2 = load_model_two()
            value_2 = transform_1(model_2, data_2)
            if value_2 < -0.00001:
                st.write('Not OAC')
            else:
                st.write('OAC')

        
        

if __name__ == "__main__":
    main()
