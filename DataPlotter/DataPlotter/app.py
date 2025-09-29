import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import io
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime
import hashlib
import json

def detect_numeric_columns(df):
    """
    Detect numeric columns excluding Date, Time, and Timestamp columns
    """
    numeric_cols = []
    for col in df.columns:
        if col not in ["Date", "Time", "Timestamp"]:
            try:
                # Sample a few values to test if they can be converted to numeric
                sample_data = df[col].dropna()
                if len(sample_data) > 0:
                    sample_size = min(10, len(sample_data))
                    pd.to_numeric(sample_data.sample(sample_size), errors="raise")
                    numeric_cols.append(col)
            except:
                pass
    return numeric_cols

def process_timestamp(df):
    """
    Process Date and Time columns to create Timestamp column
    """
    if "Date" in df.columns and "Time" in df.columns:
        # Combine Date and Time columns
        df["Timestamp"] = df["Date"].astype(str) + " " + df["Time"].astype(str)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
    else:
        # If no Date/Time columns, use index as timestamp
        df["Timestamp"] = pd.to_datetime(df.index, unit="s", errors="coerce")
    
    return df

def create_time_series_plot(df, selected_columns, chart_type="line"):
    """
    Create interactive time series plot using Plotly with different chart types
    """
    if not selected_columns:
        st.warning("Please select at least one column to plot.")
        return None
    
    fig = go.Figure()
    
    # Add traces for each selected column based on chart type
    for i, col in enumerate(selected_columns):
        # Convert column to numeric, handling errors
        numeric_data = pd.to_numeric(df[col], errors="coerce")
        
        if chart_type == "line":
            fig.add_trace(go.Scatter(
                x=df["Timestamp"],
                y=numeric_data,
                mode='lines',
                name=col,
                line=dict(width=2),
                hovertemplate=f'<b>{col}</b><br>' +
                             'Time: %{x}<br>' +
                             'Value: %{y}<br>' +
                             '<extra></extra>'
            ))
        elif chart_type == "scatter":
            fig.add_trace(go.Scatter(
                x=df["Timestamp"],
                y=numeric_data,
                mode='markers',
                name=col,
                marker=dict(size=6, opacity=0.7),
                hovertemplate=f'<b>{col}</b><br>' +
                             'Time: %{x}<br>' +
                             'Value: %{y}<br>' +
                             '<extra></extra>'
            ))
        elif chart_type == "bar":
            fig.add_trace(go.Bar(
                x=df["Timestamp"],
                y=numeric_data,
                name=col,
                opacity=0.8,
                hovertemplate=f'<b>{col}</b><br>' +
                             'Time: %{x}<br>' +
                             'Value: %{y}<br>' +
                             '<extra></extra>'
            ))
        elif chart_type == "area":
            fig.add_trace(go.Scatter(
                x=df["Timestamp"],
                y=numeric_data,
                mode='lines',
                name=col,
                fill='tonexty',
                stackgroup='one',
                line=dict(width=1),
                fillcolor=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] + '40',
                hovertemplate=f'<b>{col}</b><br>' +
                             'Time: %{x}<br>' +
                             'Value: %{y}<br>' +
                             '<extra></extra>'
            ))
    
    # Update layout based on chart type
    chart_titles = {
        "line": "Time Series Data (Line Chart)",
        "scatter": "Time Series Data (Scatter Plot)", 
        "bar": "Time Series Data (Bar Chart)",
        "area": "Time Series Data (Area Chart)"
    }
    
    # Configure layout based on chart type
    layout_updates = {
        'title': chart_titles.get(chart_type, "Time Series Data"),
        'xaxis_title': "Time",
        'yaxis_title': "Value",
        'showlegend': True,
        'height': 600,
        'xaxis': dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        'yaxis': dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        'plot_bgcolor': 'white'
    }
    
    # Chart-specific layout adjustments
    if chart_type == "bar":
        layout_updates.update({
            'barmode': 'group',
            'bargap': 0.15,
            'bargroupgap': 0.05,
            'hovermode': 'x'
        })
    else:
        layout_updates['hovermode'] = 'x unified'
    
    fig.update_layout(**layout_updates)
    
    return fig

def main():
    st.set_page_config(
        page_title="CSV Time Series Visualizer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 CSV Time Series Visualizer")
    st.write("Upload your CSV logging data and convert it into interactive time series graphs")
    
    # File upload section
    st.header("📁 Upload CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your CSV file with logging data. The app will automatically detect Date/Time columns and numeric data columns."
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file with latin1 encoding
            df = pd.read_csv(uploaded_file, encoding="latin1")
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Show raw data preview
            with st.expander("🔍 Preview Raw Data", expanded=False):
                st.dataframe(df.head(10))
            
            # Process timestamps
            df = process_timestamp(df)
            
            # Check if timestamp processing was successful
            if bool(df["Timestamp"].isna().all()):
                st.warning("⚠️ Could not parse timestamp data. Using row index as time axis.")
                df["Timestamp"] = pd.to_datetime(pd.date_range(start='2024-01-01', periods=len(df), freq='1min'))
            
            # Detect numeric columns
            numeric_cols = detect_numeric_columns(df)
            
            if not numeric_cols:
                st.error("❌ No numeric columns found in the dataset. Please check your CSV file.")
                return
            
            st.success(f"✅ Found {len(numeric_cols)} numeric columns: {', '.join(numeric_cols)}")
            
            # Column selection section
            st.header("📈 Select Data Columns and Chart Type")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_columns = st.multiselect(
                    "Choose columns to visualize:",
                    options=numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
                    help="Select one or more numeric columns to plot on the time series chart"
                )
                
                # Chart type selection
                chart_type = st.selectbox(
                    "Choose chart type:",
                    options=["line", "scatter", "bar", "area"],
                    format_func=lambda x: {
                        "line": "📈 Line Chart",
                        "scatter": "⚫ Scatter Plot", 
                        "bar": "📊 Bar Chart",
                        "area": "📈 Area Chart"
                    }[x],
                    help="Select the type of chart to visualize your data"
                )
            
            with col2:
                st.write("**Available columns:**")
                for col in numeric_cols:
                    non_null_count = df[col].notna().sum()
                    total_count = len(df)
                    st.write(f"• {col} ({non_null_count}/{total_count} values)")
            
            # Initialize variables
            filtered_df = df.copy()
            filtered_count = len(df)
            
            # Data filtering section
            if selected_columns:
                st.header("🔍 Data Filtering")
                
                # Get min/max values for filters
                min_date = df['Timestamp'].min()
                max_date = df['Timestamp'].max()
                
                col1_filter, col2_filter = st.columns([2, 2])
                
                with col1_filter:
                    st.subheader("📅 Date Range Filter")
                    date_range = st.date_input(
                        "Select date range:",
                        value=(min_date.date(), max_date.date()),
                        min_value=min_date.date(),
                        max_value=max_date.date(),
                        help="Filter data by date range"
                    )
                    
                    time_range = st.slider(
                        "Time range (hours):",
                        min_value=0,
                        max_value=23,
                        value=(0, 23),
                        help="Filter data by time of day"
                    )
                
                with col2_filter:
                    st.subheader("📊 Value Filters")
                    
                    # Value threshold filters for selected columns
                    value_filters = {}
                    for col in selected_columns:
                        try:
                            col_series = df[col]
                            col_numeric = pd.to_numeric(col_series, errors='coerce')
                            col_data = col_numeric.dropna()
                            
                            if len(col_data) > 0:
                                min_val = float(col_data.min())
                                max_val = float(col_data.max())
                                
                                if min_val != max_val:  # Only show slider if there's a range
                                    value_filters[col] = st.slider(
                                        f"{col} range:",
                                        min_value=min_val,
                                        max_value=max_val,
                                        value=(min_val, max_val),
                                        format="%.3f",
                                        help=f"Filter {col} values"
                                    )
                        except Exception:
                            continue  # Skip columns that can't be processed
                
                # Apply filters
                filtered_df = df.copy()
                
                # Date filter
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    date_mask = (
                        (filtered_df['Timestamp'].dt.date >= start_date) &
                        (filtered_df['Timestamp'].dt.date <= end_date)
                    )
                    filtered_df = filtered_df[date_mask]
                
                # Time of day filter
                time_mask = (
                    (filtered_df['Timestamp'].dt.hour >= time_range[0]) &
                    (filtered_df['Timestamp'].dt.hour <= time_range[1])
                )
                filtered_df = filtered_df[time_mask]
                
                # Value filters
                for col, (min_val, max_val) in value_filters.items():
                    try:
                        col_numeric = pd.to_numeric(filtered_df[col], errors='coerce')
                        value_mask = (col_numeric >= min_val) & (col_numeric <= max_val)
                        filtered_df = filtered_df[value_mask]
                    except Exception:
                        continue  # Skip filters that can't be applied
                
                # Show filtering results
                original_count = len(df)
                filtered_count = len(filtered_df)
                filter_pct = (filtered_count / original_count * 100) if original_count > 0 else 0
                
                st.info(f"📊 Filtered data: {filtered_count:,} / {original_count:,} rows ({filter_pct:.1f}%)")
                
                if filtered_count == 0:
                    st.error("❌ No data remains after applying filters. Please adjust your filter settings to see visualizations.")
                    st.info("💡 Tip: Try expanding the date range, time window, or value thresholds.")
            
            # Data preprocessing section
            preprocessed_df = filtered_df.copy()
            preprocessing_steps = []
            if selected_columns and filtered_count > 0:
                st.header("🔧 Data Preprocessing")
                
                # Create preprocessing options
                preprocessing_col1, preprocessing_col2 = st.columns([1, 1])
                
                with preprocessing_col1:
                    st.subheader("📈 Smoothing & Filtering")
                    
                    # Smoothing options
                    enable_smoothing = st.checkbox("Enable data smoothing", help="Apply smoothing to reduce noise in the data")
                    
                    if enable_smoothing:
                        smoothing_method = st.selectbox(
                            "Smoothing method:",
                            ["rolling_mean", "exponential", "savgol"],
                            format_func=lambda x: {
                                "rolling_mean": "Rolling Average",
                                "exponential": "Exponential Smoothing",
                                "savgol": "Savitzky-Golay Filter"
                            }[x],
                            help="Choose the smoothing algorithm"
                        )
                        
                        if smoothing_method == "rolling_mean":
                            window_size = st.slider("Window size (data points):", 2, 50, 5, help="Number of points to average")
                        elif smoothing_method == "exponential":
                            alpha = st.slider("Smoothing factor (alpha):", 0.01, 1.0, 0.3, help="Higher = more weight to recent data")
                        elif smoothing_method == "savgol":
                            window_size = st.slider("Window size:", 3, 51, 7, step=2, help="Must be odd number")
                            poly_order = st.slider("Polynomial order:", 1, min(6, window_size-1), 2, help="Degree of polynomial")
                    
                    # Interpolation options
                    enable_interpolation = st.checkbox("Enable missing value interpolation", help="Fill missing values in the data")
                    
                    if enable_interpolation:
                        interp_method = st.selectbox(
                            "Interpolation method:",
                            ["linear", "polynomial", "spline", "ffill", "bfill"],
                            format_func=lambda x: {
                                "linear": "Linear Interpolation",
                                "polynomial": "Polynomial Interpolation",
                                "spline": "Spline Interpolation",
                                "ffill": "Forward Fill",
                                "bfill": "Backward Fill"
                            }[x],
                            help="Method to fill missing values"
                        )
                
                with preprocessing_col2:
                    st.subheader("🔄 Resampling")
                    
                    enable_resampling = st.checkbox("Enable time resampling", help="Change the time frequency of the data")
                    
                    if enable_resampling:
                        resample_freq = st.selectbox(
                            "Resampling frequency:",
                            ["1min", "5min", "15min", "30min", "1H", "2H", "6H", "12H", "1D", "1W"],
                            format_func=lambda x: {
                                "1min": "1 Minute",
                                "5min": "5 Minutes", 
                                "15min": "15 Minutes",
                                "30min": "30 Minutes",
                                "1H": "1 Hour",
                                "2H": "2 Hours",
                                "6H": "6 Hours",
                                "12H": "12 Hours",
                                "1D": "1 Day",
                                "1W": "1 Week"
                            }[x],
                            help="New time frequency for the data"
                        )
                        
                        resample_method = st.selectbox(
                            "Aggregation method:",
                            ["mean", "median", "sum", "min", "max", "first", "last"],
                            format_func=lambda x: {
                                "mean": "Average",
                                "median": "Median",
                                "sum": "Sum",
                                "min": "Minimum",
                                "max": "Maximum",
                                "first": "First Value",
                                "last": "Last Value"
                            }[x],
                            help="How to combine values within each time period"
                        )
                
                # Apply preprocessing
                if enable_smoothing or enable_interpolation or enable_resampling:
                    st.write("**Apply preprocessing:**")
                    
                    if st.button("⚙️ Apply Preprocessing", help="Process the data with selected options"):
                        try:
                            with st.spinner("Processing data..."):
                                # Start with filtered data
                                preprocessed_df = filtered_df.copy()
                                preprocessing_steps = []
                                
                                # Apply interpolation first (fill missing values)
                                if enable_interpolation:
                                    interpolation_applied = False
                                    for col in selected_columns:
                                        try:
                                            if interp_method in ['linear', 'polynomial', 'spline']:
                                                if interp_method == 'polynomial':
                                                    preprocessed_df[col] = pd.to_numeric(preprocessed_df[col], errors='coerce').interpolate(method=interp_method, order=2)
                                                elif interp_method == 'spline':
                                                    preprocessed_df[col] = pd.to_numeric(preprocessed_df[col], errors='coerce').interpolate(method=interp_method, order=3)
                                                else:
                                                    preprocessed_df[col] = pd.to_numeric(preprocessed_df[col], errors='coerce').interpolate(method=interp_method)
                                            else:  # ffill or bfill
                                                preprocessed_df[col] = pd.to_numeric(preprocessed_df[col], errors='coerce').fillna(method=interp_method)
                                            interpolation_applied = True
                                        except Exception:
                                            continue
                                    if interpolation_applied:
                                        preprocessing_steps.append(f"Interpolation ({interp_method})")
                                
                                # Apply resampling
                                if enable_resampling:
                                    try:
                                        # Set timestamp as index for resampling
                                        resample_df = preprocessed_df.set_index('Timestamp')
                                        
                                        # Resample only numeric columns
                                        numeric_cols_data = {}
                                        for col in selected_columns:
                                            numeric_series = pd.to_numeric(resample_df[col], errors='coerce')
                                            if resample_method == 'mean':
                                                resampled = numeric_series.resample(resample_freq).mean()
                                            elif resample_method == 'median':
                                                resampled = numeric_series.resample(resample_freq).median()
                                            elif resample_method == 'sum':
                                                resampled = numeric_series.resample(resample_freq).sum()
                                            elif resample_method == 'min':
                                                resampled = numeric_series.resample(resample_freq).min()
                                            elif resample_method == 'max':
                                                resampled = numeric_series.resample(resample_freq).max()
                                            elif resample_method == 'first':
                                                resampled = numeric_series.resample(resample_freq).first()
                                            elif resample_method == 'last':
                                                resampled = numeric_series.resample(resample_freq).last()
                                            
                                            numeric_cols_data[col] = resampled
                                        
                                        # Create new dataframe with resampled data
                                        resampled_df = pd.DataFrame(numeric_cols_data)
                                        resampled_df['Timestamp'] = resampled_df.index
                                        resampled_df = resampled_df.reset_index(drop=True)
                                        preprocessed_df = resampled_df
                                        preprocessing_steps.append(f"Resampling ({resample_freq}, {resample_method})")
                                        
                                    except Exception as e:
                                        st.warning(f"⚠️ Resampling failed: {str(e)}")
                                
                                # Apply smoothing last
                                if enable_smoothing:
                                    smoothing_applied = False
                                    try:
                                        from scipy.signal import savgol_filter
                                        savgol_available = True
                                    except ImportError:
                                        savgol_available = False
                                        if smoothing_method == "savgol":
                                            st.warning("⚠️ Savitzky-Golay filter requires scipy. Please install scipy or choose a different smoothing method.")
                                    
                                    for col in selected_columns:
                                        try:
                                            col_data = pd.to_numeric(preprocessed_df[col], errors='coerce')
                                            
                                            if smoothing_method == "rolling_mean":
                                                smoothed = col_data.rolling(window=window_size, center=True, min_periods=1).mean()
                                            elif smoothing_method == "exponential":
                                                smoothed = col_data.ewm(alpha=alpha).mean()
                                            elif smoothing_method == "savgol" and savgol_available:
                                                # Handle NaN values for Savitzky-Golay
                                                valid_data = col_data.dropna()
                                                if len(valid_data) >= window_size:
                                                    try:
                                                        smoothed_values = savgol_filter(valid_data, window_size, poly_order)
                                                        smoothed = col_data.copy()
                                                        smoothed.loc[valid_data.index] = smoothed_values
                                                    except Exception:
                                                        smoothed = col_data  # Fallback if filter fails
                                                else:
                                                    smoothed = col_data  # Not enough data for filter
                                            else:
                                                smoothed = col_data  # No smoothing applied
                                            
                                            preprocessed_df[col] = smoothed
                                            smoothing_applied = True
                                        except Exception:
                                            continue  # Skip columns that can't be smoothed
                                    
                                    if smoothing_applied:
                                        preprocessing_steps.append(f"Smoothing ({smoothing_method})")
                                
                                # Show preprocessing summary
                                if preprocessing_steps:
                                    st.success(f"✅ Preprocessing complete! Applied: {', '.join(preprocessing_steps)}")
                                    
                                    # Show before/after comparison
                                    original_count = len(filtered_df)
                                    processed_count = len(preprocessed_df)
                                    st.info(f"📈 Data points: {original_count} → {processed_count}")
                                else:
                                    st.warning("⚠️ No preprocessing steps were applied.")
                                    
                        except Exception as e:
                            st.error(f"❌ Preprocessing failed: {str(e)}")
                            preprocessed_df = filtered_df.copy()  # Fallback to filtered data
                
                # If no preprocessing applied, use filtered data
                if 'preprocessing_steps' not in locals() or not locals().get('preprocessing_steps'):
                    preprocessing_steps = []
            
            # Generate and display the plot
            plot_df = preprocessed_df if 'preprocessed_df' in locals() and len(preprocessed_df) > 0 else filtered_df
            if selected_columns and len(plot_df) > 0:
                st.header("📊 Time Series Visualization")
                
                # Show data info
                st.info(f"📅 Time range: {plot_df['Timestamp'].min()} to {plot_df['Timestamp'].max()}")
                
                # Show preprocessing info if applied
                if 'preprocessing_steps' in locals() and preprocessing_steps:
                    st.info(f"⚙️ Preprocessing applied: {', '.join(preprocessing_steps)}")
                
                # Create the plot
                fig = create_time_series_plot(plot_df, selected_columns, chart_type)
                
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Chart Export Options
                    with st.expander("📎 Export Chart", expanded=False):
                        # Generate a hash of the figure to use as session state key
                        fig_dict = fig.to_dict()
                        fig_hash = hashlib.md5(json.dumps(fig_dict, sort_keys=True, default=str).encode()).hexdigest()[:8]
                        
                        # Initialize session state for exports
                        if 'chart_exports' not in st.session_state:
                            st.session_state.chart_exports = {}
                        
                        # Check if we already have exports for this figure
                        exports_key = f"exports_{fig_hash}"
                        if exports_key not in st.session_state.chart_exports:
                            st.session_state.chart_exports[exports_key] = {}
                        
                        current_exports = st.session_state.chart_exports[exports_key]
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        st.write("**Generate and download chart in different formats:**")
                        st.info("💡 **Tip:** Plotly's chart toolbar (top-right) also provides PNG download. Use the buttons below for SVG and PDF formats.")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # PNG Export
                            if st.button("🖼️ Generate PNG", help="Generate PNG image for download"):
                                try:
                                    with st.spinner("Generating PNG..."):
                                        img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
                                        current_exports['png'] = {
                                            'data': img_bytes,
                                            'filename': f"time_series_chart_{timestamp}.png"
                                        }
                                    st.success("✅ PNG generated successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ PNG generation failed: {str(e)}")
                                    st.write("💡 **Troubleshooting:** Make sure kaleido is installed for image export.")
                            
                            # Show PNG download button if available
                            if 'png' in current_exports:
                                st.download_button(
                                    label="📎 Download PNG",
                                    data=current_exports['png']['data'],
                                    file_name=current_exports['png']['filename'],
                                    mime="image/png"
                                )
                        
                        with col2:
                            # SVG Export
                            if st.button("🎨 Generate SVG", help="Generate scalable vector graphics for download"):
                                try:
                                    with st.spinner("Generating SVG..."):
                                        svg_bytes = fig.to_image(format="svg", width=1200, height=600)
                                        current_exports['svg'] = {
                                            'data': svg_bytes,
                                            'filename': f"time_series_chart_{timestamp}.svg"
                                        }
                                    st.success("✅ SVG generated successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ SVG generation failed: {str(e)}")
                                    st.write("💡 **Troubleshooting:** Make sure kaleido is installed for image export.")
                            
                            # Show SVG download button if available
                            if 'svg' in current_exports:
                                st.download_button(
                                    label="📎 Download SVG",
                                    data=current_exports['svg']['data'],
                                    file_name=current_exports['svg']['filename'],
                                    mime="image/svg+xml"
                                )
                        
                        with col3:
                            # PDF Export
                            if st.button("📄 Generate PDF", help="Generate PDF document for download"):
                                try:
                                    with st.spinner("Generating PDF..."):
                                        pdf_bytes = fig.to_image(format="pdf", width=1200, height=600)
                                        current_exports['pdf'] = {
                                            'data': pdf_bytes,
                                            'filename': f"time_series_chart_{timestamp}.pdf"
                                        }
                                    st.success("✅ PDF generated successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ PDF generation failed: {str(e)}")
                                    st.write("💡 **Troubleshooting:** Make sure kaleido is installed for image export.")
                            
                            # Show PDF download button if available
                            if 'pdf' in current_exports:
                                st.download_button(
                                    label="📎 Download PDF",
                                    data=current_exports['pdf']['data'],
                                    file_name=current_exports['pdf']['filename'],
                                    mime="application/pdf"
                                )
                        
                        # Show available downloads summary
                        if current_exports:
                            available_formats = list(current_exports.keys())
                            st.info(f"📁 Ready for download: {', '.join(format.upper() for format in available_formats)}")
                        
                        st.write("**Format Comparison:**")
                        st.write("• **PNG**: High-quality raster image, good for presentations and documents")
                        st.write("• **SVG**: Scalable vector graphics, perfect for web and print at any size")
                        st.write("• **PDF**: Document format, ideal for reports and professional documentation")
                    
                    # Show statistics
                    with st.expander("📊 Data Statistics", expanded=False):
                        stats_df = filtered_df[selected_columns].describe()  # Only numeric columns
                        st.dataframe(stats_df)
                    
                    # Advanced Statistical Analysis
                    with st.expander("🔍 Advanced Statistical Analysis", expanded=False):
                        analysis_col1, analysis_col2 = st.columns([1, 1])
                        
                        with analysis_col1:
                            st.subheader("📈 Trend Analysis")
                            
                            # Sort data by timestamp for chronological analysis
                            sorted_df = filtered_df.sort_values('Timestamp')
                            
                            # Trend analysis for each selected column
                            for col in selected_columns:
                                try:
                                    # Get numeric data and align with sorted timestamps
                                    col_series = pd.to_numeric(sorted_df[col], errors='coerce')
                                    valid_mask = col_series.notna()
                                    col_data = col_series[valid_mask]
                                    
                                    if len(col_data) > 5:  # Need minimum data points for analysis
                                        # Prepare time data (convert to numeric for regression)
                                        time_data = sorted_df.loc[valid_mask, 'Timestamp']
                                        
                                        # Check for degenerate time data
                                        if time_data.nunique() < 2:
                                            st.write(f"**{col}:** Unable to analyze trend - insufficient time variation")
                                            continue
                                        
                                        time_numeric = (time_data - time_data.min()).dt.total_seconds() / 3600  # Hours since start
                                        
                                        # Linear regression
                                        X = time_numeric.values.reshape(-1, 1)
                                        y = col_data.values
                                        
                                        model = LinearRegression()
                                        model.fit(X, y)
                                        r2 = r2_score(y, model.predict(X))
                                        
                                        # Calculate trend statistics
                                        slope = model.coef_[0]
                                        trend_direction = "📈 Increasing" if slope > 0 else "📉 Decreasing" if slope < 0 else "➡️ Stable"
                                        
                                        # Growth rate using chronologically first and last values
                                        first_val = col_data.iloc[0]
                                        last_val = col_data.iloc[-1]
                                        if first_val != 0:
                                            growth_rate = ((last_val - first_val) / abs(first_val)) * 100
                                        else:
                                            growth_rate = 0
                                        
                                        # Time span for context
                                        time_span = (time_data.max() - time_data.min()).total_seconds() / 3600
                                        
                                        st.write(f"**{col}:**")
                                        st.write(f"• Trend: {trend_direction}")
                                        st.write(f"• Slope: {slope:.4f} units/hour")
                                        st.write(f"• R² Score: {r2:.3f}")
                                        st.write(f"• Total Growth: {growth_rate:+.1f}% over {time_span:.1f} hours")
                                        st.write("---")
                                    else:
                                        st.write(f"**{col}:** Insufficient data points for trend analysis")
                                        
                                except Exception as e:
                                    st.write(f"**{col}:** Unable to analyze trend - {str(e)}")
                        
                        with analysis_col2:
                            st.subheader("🔢 Correlation Analysis")
                            
                            if len(selected_columns) > 1:
                                try:
                                    # Calculate correlation matrix
                                    numeric_data = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
                                    corr_matrix = numeric_data.corr()
                                    
                                    # Display correlation matrix
                                    st.write("**Correlation Matrix:**")
                                    st.dataframe(corr_matrix.round(3))
                                    
                                    # Find strongest correlations
                                    st.write("**Strong Correlations (|r| > 0.7):**")
                                    strong_correlations = []
                                    for i in range(len(corr_matrix.columns)):
                                        for j in range(i+1, len(corr_matrix.columns)):
                                            corr_val = corr_matrix.iloc[i, j]
                                            if abs(corr_val) > 0.7:
                                                col1 = corr_matrix.columns[i]
                                                col2 = corr_matrix.columns[j]
                                                strong_correlations.append((col1, col2, corr_val))
                                    
                                    if strong_correlations:
                                        for col1, col2, corr_val in strong_correlations:
                                            correlation_type = "📈 Positive" if corr_val > 0 else "📉 Negative"
                                            st.write(f"• {col1} ↔ {col2}: {corr_val:.3f} ({correlation_type})")
                                    else:
                                        st.write("No strong correlations found (|r| > 0.7)")
                                        
                                except Exception as e:
                                    st.write("Unable to calculate correlations")
                            else:
                                st.write("Select multiple columns to see correlations")
                            
                            st.subheader("📊 Statistical Tests")
                            
                            # Statistical tests for each column
                            for col in selected_columns:
                                try:
                                    col_data = pd.to_numeric(filtered_df[col], errors='coerce').dropna()
                                    if len(col_data) > 3:
                                        st.write(f"**{col}:**")
                                        
                                        # Always compute skewness and kurtosis
                                        skewness = stats.skew(col_data)
                                        kurtosis = stats.kurtosis(col_data)
                                        st.write(f"• Skewness: {skewness:.3f}")
                                        st.write(f"• Kurtosis: {kurtosis:.3f}")
                                        
                                        # Normality test - choose appropriate test based on sample size
                                        if col_data.nunique() < 2:  # Constant data
                                            st.write("• Normality: Test not applicable (constant data)")
                                        elif len(col_data) <= 5000:
                                            # Use Shapiro-Wilk for smaller samples
                                            stat, p_value = stats.shapiro(col_data)
                                            normal = "✅ Normal" if p_value > 0.05 else "❌ Non-normal"
                                            st.write(f"• Normality (Shapiro-Wilk): {normal} (p={p_value:.3f})")
                                        elif len(col_data) > 20:
                                            # Use D'Agostino-Pearson for larger samples
                                            stat, p_value = stats.normaltest(col_data)
                                            normal = "✅ Normal" if p_value > 0.05 else "❌ Non-normal"
                                            st.write(f"• Normality (D'Agostino-Pearson): {normal} (p={p_value:.3f})")
                                        else:
                                            st.write("• Normality: Sample too small for reliable test")
                                        
                                        st.write("---")
                                    else:
                                        st.write(f"**{col}:** Insufficient data for statistical tests")
                                        
                                except Exception as e:
                                    st.write(f"**{col}:** Unable to perform statistical tests - {str(e)}")
                    
                    # Download processed data option
                    with st.expander("💾 Download Processed Data", expanded=False):
                        csv_buffer = io.StringIO()
                        processed_df = plot_df[["Timestamp"] + selected_columns].copy()
                        processed_df.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv_buffer.getvalue(),
                            file_name="processed_time_series_data.csv",
                            mime="text/csv"
                        )
            else:
                st.info("👆 Please select at least one column to visualize the data.")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.write("Please check that your CSV file is properly formatted and try again.")
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a CSV file to get started.")
        
        with st.expander("ℹ️ How to use this app", expanded=True):
            st.write("""
            **Steps to visualize your CSV data:**
            
            1. **Upload CSV File**: Click the file uploader above and select your CSV file
            2. **Automatic Processing**: The app will automatically:
               - Detect Date and Time columns and combine them into timestamps
               - Find all numeric columns in your data
               - Handle encoding issues (supports latin1 encoding)
            3. **Select Columns**: Choose which numeric columns you want to plot
            4. **View Interactive Chart**: Explore your time series data with zoom, pan, and hover features
            
            **CSV File Requirements:**
            - Should contain numeric data columns for plotting
            - Optional: 'Date' and 'Time' columns for timestamp creation
            - Supports latin1 encoding for special characters
            
            **Features:**
            - Interactive time series charts with Plotly (line, scatter, bar, area)
            - Multiple parameter selection with chart type options
            - Advanced data filtering (date range, time of day, value thresholds)
            - Comprehensive statistical analysis (linear trends, correlations, normality tests, skewness, kurtosis)
            - Professional chart export (PNG, SVG, PDF formats with persistent downloads)
            - Data statistics and preview
            - Download processed data
            """)

if __name__ == "__main__":
    main()
