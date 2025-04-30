# import streamlit as st
# import pandas as pd
# import json
# import os
# import csv
# import warnings

# warnings.filterwarnings("ignore")

# def load_time_series_data():
#     """
#     Create a Streamlit file uploader with improved memory handling
#     """
#     st.header("Upload Time Series Data")
    
#     # Simple uploader with limited file types - focus on most common formats first
#     uploaded_file = st.file_uploader(
#         "Choose a file", 
#         type=["csv", "xlsx", "json", "tsv" ]
#     )
    
#     if uploaded_file is not None:
#         file_name = uploaded_file.name
#         file_extension = os.path.splitext(file_name)[1].lower()
        
#         try:
#             # Process based on file extension
#             if file_extension in ['.csv']:
#                 sniffer = csv.Sniffer()

#                 try:
#                     detected_delimiter = sniffer.sniff(uploaded_file).delimiter
#                     if detected_delimiter not in [",", ";", "\t", "|"]:
#                         detected_delimiter = ","  # Default fallback
#                 except:
#                     detected_delimiter = ","  # Default fallback if detection fails
                    
#                     try:
#                         # This reads the file only once
#                         data = pd.read_csv(uploaded_file, sep=detected_delimiter, low_memory=False, parse_dates=True)
                        
#                         datetime_candidates = []
#                         for col in data.columns:
#                             # Check if column name suggests it's a date
#                             if any(keyword in col.lower() for keyword in ['date', 'time', 'datetime', 'timestamp']):
#                                 try: 
#                                     formats = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%d-%b-%Y"]
#                                     for fmt in formats:
#                                         try:
#                                             test_conversion = pd.to_datetime(data[col], format=fmt, errors='raise')  # Raise error if format fails
#                                             break  # If successful, exit loop
#                                         except ValueError:
#                                             continue  # Try the next format

#                                     # If all formats fail, use the fallback method
#                                     if test_conversion.isna().all():
#                                         test_conversion = pd.to_datetime(data[col], errors='coerce')  # Fallback with warning suppressed
                                    
#                                     # Check if conversion was successful for most values
#                                     if test_conversion.notna().sum() / len(data) > 0.5:
#                                         datetime_candidates.append(col)
#                                 except:
#                                     pass
                    
#                         # If datetime candidates found
#                         if datetime_candidates:
#                             # If multiple candidates, let user choose
#                             if len(datetime_candidates) > 1:
#                                 datetime_col = st.selectbox(
#                                     "Multiple date columns detected. Select the primary datetime column:", 
#                                     options=datetime_candidates
#                                 )
#                             else:
#                                 datetime_col = datetime_candidates[0]
                            
#                             # Ensure the selected column is parsed as datetime
#                             data[datetime_col] = pd.to_datetime(data[datetime_col], errors='coerce')
                            
#                             # Sort by datetime column
#                             data = data.sort_values(by=datetime_col)
#                         else:
#                             # If no clear datetime column, warn the user
#                             st.warning("No datetime column automatically detected. Please select one manually.")
#                             datetime_col = None
                        
#                         # Show a small preview
#                         st.dataframe(data.head(15))
#                         # df = data.copy()
                        
#                         # If a datetime column was found or selected, return it along with the data
#                         return data, 'csv', datetime_col
                        
#                     except Exception as e:
#                         st.error(f"Error loading CSV: {str(e)}")
                    
#             elif file_extension in ['.xlsx', '.xls']:
#                 try:
#                     # Read directly without showing preview first
#                     data = pd.read_excel(uploaded_file)
                    
#                      # Detect potential datetime columns (similar to CSV logic)
#                     datetime_candidates = []
#                     for col in data.columns:
#                         if any(keyword in col.lower() for keyword in ['date', 'time', 'datetime', 'timestamp']):
#                             try:
#                                 formats = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"]
#                                 for fmt in formats:
#                                         try:
#                                             test_conversion = pd.to_datetime(data[col], format=fmt, errors='raise')  # Raise error if format fails
#                                             break  # If successful, exit loop
#                                         except ValueError:
#                                             continue  # Try the next format

#                                     # If all formats fail, use the fallback method
#                                 if test_conversion.isna().all():
#                                     test_conversion = pd.to_datetime(data[col], errors='coerce')  # Fallback with warning suppressed
                                
#                                 # Check if conversion was successful for most values
#                                 if test_conversion.notna().sum() / len(data) > 0.5:
#                                     datetime_candidates.append(col)
#                             except:
#                                 pass
                    
#                     # If datetime candidates found
#                     if datetime_candidates:
#                         # If multiple candidates, let user choose
#                         if len(datetime_candidates) > 1:
#                             datetime_col = st.selectbox(
#                                 "Multiple date columns detected. Select the primary datetime column:", 
#                                 options=datetime_candidates
#                             )
#                         else:
#                             datetime_col = datetime_candidates[0]
                        
#                         # Ensure the selected column is parsed as datetime
#                         data[datetime_col] = pd.to_datetime(data[datetime_col], errors='coerce')
                        
#                         # Sort by datetime column
#                         data = data.sort_values(by=datetime_col)
#                     else:
#                         # If no clear datetime column, warn the user
#                         st.warning("No datetime column automatically detected. Please select one manually.")
#                         datetime_col = None
                    
#                     # Show a small preview of the data
#                     st.dataframe(data.head(5))
#                     # df = data.copy()
                    
#                     return data, 'excel', datetime_col

                    
#                 except Exception as e:
#                     st.error(f"Error loading Excel file: {str(e)}")
                    
#             elif file_extension in ['.json']:
#                 try:
#                     # First parse the JSON
#                     json_content = uploaded_file.getvalue().decode('utf-8')
#                     json_data = json.loads(json_content)
                    
#                      # Try different approaches to convert JSON to DataFrame
#                     try:
#                         # Method 1: Direct conversion if it's a simple array
#                         if isinstance(json_data, list) and len(json_data) > 0:
#                             st.info("Converting JSON array to DataFrame")
#                             data = pd.DataFrame(json_data)
                            
#                         # Method 2: Find and use the first array in the structure
#                         elif isinstance(json_data, dict):
#                             # Option for user to select which part of JSON to use
#                             st.subheader("JSON Structure Options")
                            
#                             # Find all possible paths that lead to arrays
#                             array_paths = []
                            
#                             def find_arrays(obj, current_path=""):
#                                 if isinstance(obj, dict):
#                                     for key, value in obj.items():
#                                         path = f"{current_path}.{key}" if current_path else key
#                                         if isinstance(value, list) and len(value) > 0:
#                                             array_paths.append((path, value))
#                                         find_arrays(value, path)
#                                 elif isinstance(obj, list):
#                                     for i, item in enumerate(obj):
#                                         find_arrays(item, f"{current_path}[{i}]")
                            
#                             find_arrays(json_data)
                            
#                             if array_paths:
#                                 path_options = [p[0] for p in array_paths]
#                                 selected_path = st.selectbox(
#                                     "Select which array to use:", 
#                                     options=path_options,
#                                     help="Choose the JSON path containing your data array"
#                                 )
                                
#                                 # Get the array data for the selected path
#                                 selected_array = next(arr for path, arr in array_paths if path == selected_path)
                                
#                                 # Convert to DataFrame
#                                 if isinstance(selected_array[0], dict):
#                                     data = pd.DataFrame(selected_array)
#                                 else:
#                                     st.error("Selected array doesn't contain objects/dictionaries that can be converted to rows")
#                                     # Show the data for debugging
#                                     st.json(selected_array[:5])  # Show limited preview
#                                     return None, None
#                             else:
#                                 # Last resort - flatten the whole structure
#                                 st.info("No arrays found. Attempting to normalize entire JSON structure.")
#                                 data = pd.json_normalize(json_data)
#                         else:
#                             st.error("JSON format not recognized")
#                             return None, None
                        
#                         # Show a preview of the data
#                         st.dataframe(data.head(5))
#                         # df = data.copy()
#                         return data, 'json', datetime_col
                            
#                     except Exception as e:
#                         st.error(f"Could not convert JSON to DataFrame: {str(e)}")
#                         # Show the raw JSON for debugging
#                         st.subheader("Raw JSON Data")
#                         st.json(json_data)
                        
#                 except Exception as e:
#                     st.error(f"Failed to parse JSON: {str(e)}")
                
#             elif file_extension in ['.tsv']:
#                 try:
#                     # Read directly without showing preview first
#                     data = pd.read_csv(uploaded_file)
                    
#                      # Detect potential datetime columns (similar to CSV logic)
#                     datetime_candidates = []
#                     for col in data.columns:
#                         if any(keyword in col.lower() for keyword in ['date', 'time', 'datetime', 'timestamp']):
#                             try:
#                                 formats = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"]
#                                 for fmt in formats:
#                                         try:
#                                             test_conversion = pd.to_datetime(data[col], format=fmt, errors='raise')  # Raise error if format fails
#                                             break  # If successful, exit loop
#                                         except ValueError:
#                                             continue  # Try the next format

#                                     # If all formats fail, use the fallback method
#                                 if test_conversion.isna().all():
#                                     test_conversion = pd.to_datetime(data[col], errors='coerce')  # Fallback with warning suppressed
                                
#                                 # Check if conversion was successful for most values
#                                 if test_conversion.notna().sum() / len(data) > 0.5:
#                                     datetime_candidates.append(col)
#                             except:
#                                 pass
                    
#                     # If datetime candidates found
#                     if datetime_candidates:
#                         # If multiple candidates, let user choose
#                         if len(datetime_candidates) > 1:
#                             datetime_col = st.selectbox(
#                                 "Multiple date columns detected. Select the primary datetime column:", 
#                                 options=datetime_candidates
#                             )
#                         else:
#                             datetime_col = datetime_candidates[0]
                        
#                         # Ensure the selected column is parsed as datetime
#                         data[datetime_col] = pd.to_datetime(data[datetime_col], errors='coerce')
                        
#                         # Sort by datetime column
#                         data = data.sort_values(by=datetime_col)
#                     else:
#                         # If no clear datetime column, warn the user
#                         st.warning("No datetime column automatically detected. Please select one manually.")
#                         datetime_col = None
                    
#                     # Only show a small preview
#                     st.success(f"Successfully loaded TSV file with {len(data)} rows and {len(data.columns)} columns")
#                     st.dataframe(data.head(5))
                        
#                     # df = data.copy()
#                     return data, 'tsv', datetime_col
                        
#                 except Exception as e:
#                     st.error(f"Error loading TSV file: {str(e)}")
                    
#             else:
#                 st.error(f"Unsupported file format: {file_extension}")
                
#         except Exception as e:
#             st.error(f"Error processing file: {str(e)}")
#     # If we get here, either no file was uploaded or processing didn't complete successfully
#     return None, None, None

import streamlit as st
import pandas as pd
import json
import os
import warnings
from io import StringIO

warnings.filterwarnings("ignore")

def load_time_series_data():
    """
    Uploads a time series file, auto-detects date columns, 
    and always converts the final DataFrame to a CSV-like DataFrame.
    """
    st.header("Upload Time Series Data")
    
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=["csv", "xlsx", "json", "tsv"]
    )
    
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_extension = os.path.splitext(file_name)[1].lower()
        
        try:
            data = None  # Initialize
            datetime_col = None  # Initialize
            
            # Read the uploaded file
            if file_extension == '.csv' or file_extension == '.tsv':
                delimiter = ',' if file_extension == '.csv' else '\t'
                data = pd.read_csv(uploaded_file, delimiter=delimiter, low_memory=False)
                
            elif file_extension in ['.xlsx', '.xls']:
                data = pd.read_excel(uploaded_file)
                
            elif file_extension == '.json':
                json_content = uploaded_file.getvalue().decode('utf-8')
                json_data = json.loads(json_content)
                
                if isinstance(json_data, list):
                    data = pd.DataFrame(json_data)
                elif isinstance(json_data, dict):
                    try:
                        data = pd.json_normalize(json_data)
                    except Exception as e:
                        st.error(f"Could not normalize JSON: {str(e)}")
                        return None, None, None
                else:
                    st.error("Unsupported JSON structure")
                    return None, None, None

            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None, None, None

            # Attempt to detect datetime columns
            datetime_candidates = []
            for col in data.columns:
                if any(keyword in col.lower() for keyword in ['date', 'time', 'datetime', 'timestamp']):
                    try:
                        formats = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%Y/%m/%d", "%d-%b-%Y", "%Y-%m-%d %H:%M:%S"]
                        for fmt in formats:
                            try:
                                test_conversion = pd.to_datetime(data[col], format=fmt, errors='raise')
                                break
                            except ValueError:
                                continue
                        if test_conversion.isna().all():
                            test_conversion = pd.to_datetime(data[col], errors='coerce')
                        
                        if test_conversion.notna().sum() / len(data) > 0.5:
                            datetime_candidates.append(col)
                    except:
                        pass

            if datetime_candidates:
                if len(datetime_candidates) > 1:
                    datetime_col = st.selectbox(
                        "Multiple datetime columns detected. Select one:", 
                        options=datetime_candidates
                    )
                else:
                    datetime_col = datetime_candidates[0]

                data[datetime_col] = pd.to_datetime(data[datetime_col], errors='coerce')
                data = data.sort_values(by=datetime_col)
            else:
                st.warning("No datetime column detected automatically.")
            
            # Now, re-convert the final DataFrame to CSV-style
            csv_buffer = StringIO()
            data.to_csv(csv_buffer, index=False)
            csv_data = pd.read_csv(StringIO(csv_buffer.getvalue()))  # Reload it as CSV structure
            st.dataframe(csv_data.head(10))

            return csv_data, 'csv', datetime_col
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    return None, None, None
