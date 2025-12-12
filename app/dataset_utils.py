#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ###########################################################################
# Dataset loading and management utilities
#
# @author      James R. Heselden (github: Iranaphor)
# @maintainer  James R. Heselden (github: Iranaphor)
# @datecreated 12th December 2025
# @datemodified 12th December 2025
# @credits     Developed with assistance from Claude Sonnet 4.5 and GitHub
#              Copilot.
# ###########################################################################

import yaml
import numpy as np


def load_dataset_yaml(yaml_path):
    """Load dataset YAML containing obstacle definitions and RF signatures."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    obstacles = data.get('obstacles', [])
    signatures = data.get('signatures', {})
    
    print(f"[DATASET] Loaded {len(obstacles)} obstacle types")
    print(f"[DATASET] Loaded {len(signatures)} existing signatures")
    
    return obstacles, signatures


def save_dataset_yaml(yaml_path, obstacles, signatures):
    """Save dataset YAML with obstacle definitions and RF signatures."""
    # Clean up signature keys - remove any whitespace or newlines
    cleaned_signatures = {}
    for key, value in signatures.items():
        # Ensure signature keys have no whitespace/newlines
        clean_key = key.strip().replace('\n', '').replace('\r', '')
        cleaned_signatures[clean_key] = value
    
    # Create custom dumper class for flow-style lists
    class CustomDumper(yaml.SafeDumper):
        pass
    
    def represent_list_flow(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    
    CustomDumper.add_representer(list, represent_list_flow)
    
    with open(yaml_path, 'w') as f:
        # Write obstacles section with each obstacle on its own line
        f.write('obstacles:\n')
        for obstacle in obstacles:
            f.write(f'  - {obstacle}\n')
        
        # Write signatures section manually to avoid "?" complex key syntax
        f.write('signatures:\n')
        for sig_key, sig_data in cleaned_signatures.items():
            # Write signature key directly (no "?" prefix)
            f.write(f'  {sig_key}:\n')
            
            # Write bins array as single-line flow style
            bins = sig_data.get('bins', [])
            bins_str = '[' + ', '.join(str(b) for b in bins) + ']'
            f.write(f'    bins: {bins_str}\n')
            
            # Write each RF type array as single-line flow style
            for rf_key, rf_values in sig_data.items():
                if rf_key == 'bins':
                    continue
                
                # Convert values to YAML-compatible format
                rf_lines = []
                for row in rf_values:
                    row_strs = []
                    for val in row:
                        if val is None:
                            row_strs.append('null')
                        else:
                            # Round to 1 decimal place and remove negative sign
                            abs_val = abs(round(float(val), 1))
                            row_strs.append(str(abs_val))
                    rf_lines.append('[' + ', '.join(row_strs) + ']')
                
                # Write as flow-style array of arrays (single line per row)
                f.write(f'    {rf_key}:\n')
                for row_str in rf_lines:
                    f.write(f'      - {row_str}\n')
    
    print(f"[DATASET] Saved {len(signatures)} signatures to {yaml_path}")


def build_obstacle_type_map(obstacles):
    """Build mapping from obstacle type to signature tag."""
    type_to_tag = {}
    tag_to_type = {}
    
    for obs in obstacles:
        obs_type = obs['type']
        tag = str(obs['signature_tag'])
        type_to_tag[obs_type] = tag
        tag_to_type[tag] = obs_type
    
    return type_to_tag, tag_to_type


def obstacle_signature_to_string(obstacle_grid, type_to_tag):
    """Convert obstacle grid row to signature string using tags."""
    n_r = obstacle_grid.shape[0]
    sig_chars = []
    
    for ir in range(n_r):
        cat = obstacle_grid[ir]
        tag = type_to_tag.get(cat, type_to_tag.get('unknown', '0'))
        sig_chars.append(tag)
    
    return 's' + ''.join(sig_chars)


def extract_rf_data_by_signature(obstacle_grid, heatmap, type_to_tag):
    """
    Extract RF data organized by obstacle signature.
    Returns dict mapping signature strings to lists of RF value arrays.
    """
    n_theta, n_r = obstacle_grid.shape
    
    sig_to_rf_data = {}
    
    for ith in range(n_theta):
        sig_str = obstacle_signature_to_string(obstacle_grid[ith, :], type_to_tag)
        rf_values = heatmap[ith, :]
        
        if sig_str not in sig_to_rf_data:
            sig_to_rf_data[sig_str] = []
        
        sig_to_rf_data[sig_str].append(rf_values)
    
    return sig_to_rf_data


def extend_signature_data(existing_sig_data, new_rf_arrays, new_bins, rf_type='ssid-espnow'):
    """
    Extend existing signature data with new RF measurements.
    Handles bin alignment and NaN padding.
    """
    if existing_sig_data is None:
        existing_sig_data = {'bins': [], rf_type: []}
    
    existing_bins = existing_sig_data.get('bins', [])
    existing_rf = existing_sig_data.get(rf_type, [])
    
    # If no existing data, just use new data
    if not existing_bins:
        result_bins = list(new_bins)
        result_rf = []
        for arr in new_rf_arrays:
            result_rf.append([float(v) if not np.isnan(v) else None for v in arr])
        return {'bins': result_bins, rf_type: result_rf}
    
    # Merge bins
    all_bins = sorted(set(existing_bins + list(new_bins)))
    n_bins = len(all_bins)
    
    # Create index mappings
    existing_bin_to_idx = {b: i for i, b in enumerate(existing_bins)}
    new_bin_to_idx = {b: i for i, b in enumerate(new_bins)}
    result_bin_to_idx = {b: i for i, b in enumerate(all_bins)}
    
    # Extend existing RF data
    extended_existing = []
    for rf_row in existing_rf:
        new_row = [None] * n_bins
        for old_idx, val in enumerate(rf_row):
            old_bin = existing_bins[old_idx]
            new_idx = result_bin_to_idx[old_bin]
            new_row[new_idx] = val
        extended_existing.append(new_row)
    
    # Add new RF data
    for arr in new_rf_arrays:
        new_row = [None] * n_bins
        for new_idx, val in enumerate(arr):
            if new_idx >= len(new_bins):
                continue
            new_bin = new_bins[new_idx]
            result_idx = result_bin_to_idx[new_bin]
            new_row[result_idx] = float(val) if not np.isnan(val) else None
        extended_existing.append(new_row)
    
    return {'bins': all_bins, rf_type: extended_existing}


def merge_signatures(existing_signatures, new_sig_data, rf_type='ssid-espnow'):
    """
    Merge new signature data into existing dataset.
    Returns updated signatures dict.
    """
    merged = dict(existing_signatures)
    
    for sig_str, rf_arrays in new_sig_data.items():
        existing_sig = merged.get(sig_str)
        
        # Extract bins from first array
        new_bins = list(range(len(rf_arrays[0])))
        
        # Extend or create signature data
        merged_sig_data = extend_signature_data(
            existing_sig, 
            rf_arrays, 
            new_bins, 
            rf_type=rf_type
        )
        
        merged[sig_str] = merged_sig_data
    
    return merged


def convert_bins_to_distances(bins, dr_m, r_start=0):
    """Convert bin indices to distance values in meters."""
    return [r_start + b * dr_m for b in bins]


def prepare_signatures_for_yaml(signatures, radii_m):
    """
    Prepare signatures dict for YAML output with proper distance bins.
    Convert bin indices to actual distance values and clean up data.
    """
    import numpy as np
    
    prepared = {}
    
    for sig_str, sig_data in signatures.items():
        if isinstance(sig_data, dict):
            prepared_sig = {}
            
            # Get bins
            bins = sig_data.get('bins', [])
            if bins:
                # Convert to distances if needed
                if all(isinstance(b, int) and b < len(radii_m) for b in bins):
                    distances = [float(radii_m[b]) for b in bins]
                else:
                    distances = bins
                
                prepared_sig['bins'] = distances
            
            # Copy RF data for each type, cleaning up NaN/Nan/None values
            for key, value in sig_data.items():
                if key != 'bins':
                    if isinstance(value, list):
                        cleaned = []
                        for item in value:
                            if isinstance(item, list):
                                # Clean nested lists
                                cleaned_row = []
                                for v in item:
                                    if v is None or v == 'Nan' or v == 'NaN':
                                        cleaned_row.append(None)
                                    elif isinstance(v, (int, float)):
                                        if np.isnan(v):
                                            cleaned_row.append(None)
                                        else:
                                            cleaned_row.append(float(v))
                                    else:
                                        try:
                                            cleaned_row.append(float(v))
                                        except (ValueError, TypeError):
                                            cleaned_row.append(None)
                                cleaned.append(cleaned_row)
                            else:
                                cleaned.append(item)
                        prepared_sig[key] = cleaned
                    else:
                        prepared_sig[key] = value
            
            prepared[sig_str] = prepared_sig
        else:
            prepared[sig_str] = sig_data
    
    return prepared
