# Overview

This is a Streamlit-based data visualization and analysis application that specializes in time series data processing. The application allows users to upload datasets, automatically detects numeric columns, processes timestamp data, and creates interactive visualizations using Plotly. The system is designed to handle CSV data with Date/Time columns and provides statistical analysis capabilities including linear regression and correlation analysis.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit for web interface
- **Visualization**: Plotly for interactive charts and graphs
- **Chart Types**: Supports multiple visualization types including line plots, scatter plots, and subplots
- **User Interface**: File upload functionality with automatic data processing and column detection

## Data Processing Layer
- **Data Handling**: Pandas for data manipulation and analysis
- **Numeric Detection**: Automatic identification of numeric columns excluding temporal data
- **Timestamp Processing**: Intelligent handling of Date/Time columns with fallback to index-based timestamps
- **Data Validation**: Error handling for data type conversion and timestamp parsing

## Analytics Engine
- **Statistical Analysis**: SciPy for statistical computations
- **Machine Learning**: Scikit-learn for linear regression and R-squared calculations
- **Data Science**: NumPy for numerical operations and mathematical computations

## Core Design Patterns
- **Modular Functions**: Separated concerns with dedicated functions for data detection, processing, and visualization
- **Error Handling**: Graceful handling of data conversion errors and missing values
- **Data Flexibility**: Support for various date formats with dayfirst parsing option
- **Interactive Visualization**: Real-time chart generation based on user column selections

# External Dependencies

## Python Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive visualization library (graph_objects and express modules)
- **numpy**: Numerical computing
- **scipy**: Scientific computing and statistics
- **scikit-learn**: Machine learning algorithms and metrics
- **datetime**: Date and time handling
- **hashlib**: Data hashing utilities
- **json**: JSON data processing
- **io**: Input/output operations

## Data Requirements
- **Input Format**: CSV files with optional Date and Time columns
- **Timestamp Handling**: Automatic detection and processing of temporal data
- **Numeric Data**: Automatic identification of quantitative columns for analysis