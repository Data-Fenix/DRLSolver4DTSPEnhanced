import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

def expand_dataset(input_file='data.csv', output_file='data_expanded.csv', 
                   current_cities=100, target_cities=150):
    """
    Expand the dataset from current_cities to target_cities.
    
    Parameters:
    - input_file: Path to existing data.csv
    - output_file: Path to save expanded dataset
    - current_cities: Current number of cities (default 100)
    - target_cities: Target number of cities (must be > current_cities)
    """
    
    # Load existing data
    print(f"Loading existing data from {input_file}...")
    existing_data = np.loadtxt(input_file, delimiter=',')
    
    # Verify dimensions
    expected_rows = current_cities * current_cities
    if existing_data.shape[0] != expected_rows:
        print(f"Warning: Expected {expected_rows} rows, found {existing_data.shape[0]}")
    
    if existing_data.shape[1] != 12:
        print(f"Warning: Expected 12 columns, found {existing_data.shape[1]}")
    
    # Calculate statistics from existing data for synthetic generation
    mean_travel_time = np.mean(existing_data)
    std_travel_time = np.std(existing_data)
    min_travel_time = np.min(existing_data)
    max_travel_time = np.max(existing_data)
    
    print(f"\nExisting data statistics:")
    print(f"  Mean: {mean_travel_time:.6f}")
    print(f"  Std: {std_travel_time:.6f}")
    print(f"  Min: {min_travel_time:.6f}")
    print(f"  Max: {max_travel_time:.6f}")
    
    # Calculate new dimensions
    new_cities = target_cities - current_cities
    total_rows_needed = target_cities * target_cities
    new_rows_needed = total_rows_needed - existing_data.shape[0]
    
    print(f"\nExpanding from {current_cities} to {target_cities} cities...")
    print(f"  New cities to add: {new_rows_needed}")
    print(f"  Total rows needed: {total_rows_needed}")
    
    # Initialize expanded matrix
    expanded_data = np.zeros((total_rows_needed, 12))
    
    # Copy existing data
    expanded_data[:existing_data.shape[0], :] = existing_data
    print(f"  Copied {existing_data.shape[0]} existing rows")
    
    # Generate new data for additional city pairs
    print(f"  Generating {new_rows_needed} new rows...")
    
    # Strategy 1: For new city pairs, generate realistic time-dependent travel times
    # We'll create patterns similar to existing data with some variation
    
    row_idx = existing_data.shape[0]
    
    for i in range(current_cities, target_cities):
        for j in range(target_cities):
            # Generate time-dependent travel times
            # Use a base travel time with time-of-day variation
            base_time = np.random.uniform(min_travel_time, max_travel_time)
            
            # Create time-dependent pattern (simulating traffic patterns)
            # Peak hours typically have higher travel times
            time_pattern = np.array([
                1.0,   # 0:00 (midnight) - low traffic
                0.9,   # 2:00 - very low
                0.85,  # 4:00 - very low
                0.9,   # 6:00 - morning starts
                1.2,   # 8:00 - peak morning
                1.1,   # 10:00 - still high
                1.0,   # 12:00 - lunch
                1.1,   # 14:00 - afternoon
                1.3,   # 16:00 - peak evening
                1.2,   # 18:00 - still high
                1.0,   # 20:00 - evening
                0.95   # 22:00 - night
            ])
            
            # Add some randomness
            noise = np.random.normal(0, 0.1, 12)
            travel_times = base_time * time_pattern * (1 + noise)
            
            # Ensure non-negative
            travel_times = np.maximum(travel_times, 0.001)
            
            # If same city (i == j), travel time is 0
            if i == j:
                travel_times = np.zeros(12)
            
            expanded_data[row_idx, :] = travel_times
            row_idx += 1
    
    # Also need to add rows for existing cities to new cities
    for i in range(current_cities):
        for j in range(current_cities, target_cities):
            # Generate similar pattern
            base_time = np.random.uniform(min_travel_time, max_travel_time)
            time_pattern = np.array([
                1.0, 0.9, 0.85, 0.9, 1.2, 1.1, 1.0, 1.1, 1.3, 1.2, 1.0, 0.95
            ])
            noise = np.random.normal(0, 0.1, 12)
            travel_times = base_time * time_pattern * (1 + noise)
            travel_times = np.maximum(travel_times, 0.001)
            
            expanded_data[row_idx, :] = travel_times
            row_idx += 1
    
    # Save expanded dataset
    print(f"\nSaving expanded dataset to {output_file}...")
    np.savetxt(output_file, expanded_data, delimiter=',', fmt='%.6f')
    
    print(f"✓ Successfully created expanded dataset!")
    print(f"  Total rows: {expanded_data.shape[0]}")
    print(f"  Total columns: {expanded_data.shape[1]}")
    print(f"  Expected for {target_cities} cities: {target_cities * target_cities} rows")
    
    return expanded_data


# Alternative: Generate data based on distance (more realistic)
def expand_dataset_with_distance(input_file='data.csv', output_file='data_expanded.csv',
                                 current_cities=100, target_cities=150):
    """
    Alternative method: Generate travel times based on Euclidean distance
    between randomly placed cities, with time-dependent factors.
    """
    # Load existing data
    existing_data = np.loadtxt(input_file, delimiter=',')
    
    # Generate random city coordinates for new cities
    np.random.seed(42)  # For reproducibility
    new_cities_coords = np.random.rand(target_cities - current_cities, 2)
    
    # Calculate distance-based travel times
    total_rows = target_cities * target_cities
    expanded_data = np.zeros((total_rows, 12))
    
    # Copy existing data
    expanded_data[:existing_data.shape[0], :] = existing_data
    
    row_idx = existing_data.shape[0]
    
    # Time-dependent multipliers (traffic patterns)
    time_multipliers = np.array([
        1.0, 0.9, 0.85, 0.9, 1.2, 1.1, 1.0, 1.1, 1.3, 1.2, 1.0, 0.95
    ])
    
    # Generate for new city pairs
    for i in range(current_cities, target_cities):
        for j in range(target_cities):
            if i == j:
                expanded_data[row_idx, :] = np.zeros(12)
            else:
                # Calculate distance (if we had coordinates)
                # For simplicity, use random distance
                base_distance = np.random.uniform(0.01, 0.15)
                
                # Add time variation
                travel_times = base_distance * time_multipliers
                
                # Add noise
                noise = np.random.normal(0, 0.02, 12)
                travel_times = travel_times * (1 + noise)
                travel_times = np.maximum(travel_times, 0.001)
                
                expanded_data[row_idx, :] = travel_times
            
            row_idx += 1
    
    # Add rows for existing cities to new cities
    for i in range(current_cities):
        for j in range(current_cities, target_cities):
            base_distance = np.random.uniform(0.01, 0.15)
            travel_times = base_distance * time_multipliers
            noise = np.random.normal(0, 0.02, 12)
            travel_times = travel_times * (1 + noise)
            travel_times = np.maximum(travel_times, 0.001)
            
            expanded_data[row_idx, :] = travel_times
            row_idx += 1
    
    np.savetxt(output_file, expanded_data, delimiter=',', fmt='%.6f')
    print(f"✓ Expanded dataset saved to {output_file}")
    return expanded_data


if __name__ == "__main__":
    # Example usage:
    # Expand to 150 cities
    expand_dataset(
        input_file='data.csv',
        output_file='data_150cities.csv',
        current_cities=100,
        target_cities=150
    )
    
    # Or expand to 200 cities
    # expand_dataset(
    #     input_file='data.csv',
    #     output_file='data_200cities.csv',
    #     current_cities=100,
    #     target_cities=200
    # )