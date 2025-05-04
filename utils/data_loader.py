import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import datetime
from io import StringIO

def load_csv_data(uploaded_file):
    """
    Load data from an uploaded CSV file
    
    Parameters:
    -----------
    uploaded_file: StreamlitUploadedFile
        The uploaded CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the loaded data
    """
    try:
        # Check if file is empty
        if uploaded_file.size == 0:
            st.error("Uploaded file is empty.")
            return None
            
        # Try to read the file
        try:
            data = pd.read_csv(uploaded_file)
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty or contains no data.")
            return None
        except pd.errors.ParserError:
            st.error("Error parsing the CSV file. Please ensure it's a valid CSV file.")
            return None
            
        # Check if data is empty
        if data.empty:
            st.error("The loaded data is empty.")
            return None
            
        # Check for required columns
        if 'Date' not in data.columns and 'date' not in data.columns:
            st.warning("No 'Date' column found in the data. Some features may not work properly.")
            
        return data
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def fetch_yahoo_finance_data(ticker, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance
    
    Parameters:
    -----------
    ticker: str
        Stock ticker symbol
    start_date: datetime
        Start date for fetching data
    end_date: datetime
        End date for fetching data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the stock data
    """
    try:
        # Validate ticker
        if not ticker or not isinstance(ticker, str):
            st.error("Invalid ticker symbol.")
            return None
            
        # Validate dates
        if not isinstance(start_date, (datetime.date, datetime.datetime)):
            st.error("Invalid start date.")
            return None
        if not isinstance(end_date, (datetime.date, datetime.datetime)):
            st.error("Invalid end date.")
            return None
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            return None
            
        # Fetch data
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Check if data is empty
        if data.empty:
            st.error(f"No data found for ticker {ticker} in the specified date range.")
            return None
            
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Add ticker column
        data['Ticker'] = ticker
        
        return data
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {str(e)}")
        return None

def get_summary_stats(data):
    """
    Generate summary statistics for the data
    
    Parameters:
    -----------
    data: pd.DataFrame
        The input data
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics of the data
    """
    try:
        if data is None or data.empty:
            st.error("No data available for summary statistics.")
            return None
            
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            st.warning("No numeric columns found in the data.")
            return None
            
        # Generate summary statistics
        summary = numeric_data.describe()
        
        # Add additional statistics
        summary.loc['skew'] = numeric_data.skew()
        summary.loc['kurtosis'] = numeric_data.kurtosis()
        
        return summary
    except Exception as e:
        st.error(f"Error generating summary statistics: {str(e)}")
        return None
