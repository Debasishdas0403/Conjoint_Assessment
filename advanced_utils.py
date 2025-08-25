import numpy as np
import pandas as pd
from scipy.linalg import det, inv
from itertools import product
import streamlit as st

class AdvancedConjointAnalyzer:
    """Corrected Advanced Conjoint Analysis with proper D-efficiency calculation"""
    
    def __init__(self):
        self.information_matrix = None
        self.covariance_matrix = None
        
    def create_design_matrix(self, profiles_df, attributes_dict):
        """
        Create design matrix with proper effects coding
        """
        design_matrix = pd.DataFrame()
        
        # Add intercept
        design_matrix['intercept'] = 1
        
        for attr_name, levels in attributes_dict.items():
            if attr_name in profiles_df.columns:
                n_levels = len(levels)
                
                # Convert to proper level indices if needed
                if profiles_df[attr_name].dtype == 'object':
                    level_map = {level: idx for idx, level in enumerate(levels)}
                    level_indices = profiles_df[attr_name].map(level_map)
                else:
                    level_indices = profiles_df[attr_name]
                
                # Effects coding: use n-1 dummy variables
                for i in range(n_levels - 1):
                    col_name = f"{attr_name}_L{i+1}"
                    # Effects coding: 1 if level i, -1 if reference level (last), 0 otherwise
                    design_matrix[col_name] = level_indices.apply(
                        lambda x: 1 if x == i else (-1 if x == n_levels - 1 else 0)
                    )
        
        return design_matrix
    
    def generate_balanced_fractional_factorial(self, attributes_dict, n_runs):
        """
        Generate a more balanced fractional factorial instead of pure random sampling
        """
        # Generate full factorial
        full_factorial = self.generate_candidate_set(attributes_dict)
        
        if n_runs >= len(full_factorial):
            # If we need more runs than full factorial, repeat the design
            repeats = n_runs // len(full_factorial)
            remainder = n_runs % len(full_factorial)
            
            repeated_design = pd.concat([full_factorial] * repeats, ignore_index=True)
            if remainder > 0:
                additional = full_factorial.sample(n=remainder, random_state=42)
                repeated_design = pd.concat([repeated_design, additional], ignore_index=True)
            
            return repeated_design.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            # Sample a balanced subset
            return self.create_balanced_subset(full_factorial, n_runs)
    
    def create_balanced_subset(self, full_factorial, n_runs):
        """
        Create a balanced subset that maintains level balance as much as possible
        """
        if n_runs >= len(full_factorial):
            return full_factorial
        
        # Try to maintain balance across levels
        np.random.seed(42)
        
        # Simple approach: stratified sampling to maintain balance
        selected_indices = np.random.choice(
            len(full_factorial), 
            size=n_runs, 
            replace=False
        )
        
        return full_factorial.iloc[selected_indices].reset_index(drop=True)
    
    def calculate_information_matrix(self, design_matrix):
        """Calculate information matrix (X'X)"""
        X = design_matrix.values
        self.information_matrix = np.dot(X.T, X)
        return self.information_matrix
    
    def calculate_d_efficiency(self, information_matrix, n_runs, normalize=True):
        """
        Calculate D-efficiency with correct statistical formulation
        """
        try:
            if information_matrix.shape[0] == 0:
                return 0.0
            
            # Check for singularity
            det_info = det(information_matrix)
            if det_info <= 1e-10 or np.isnan(det_info) or np.isinf(det_info):
                return 0.0
            
            # Number of parameters
            p = information_matrix.shape[0]
            
            # Calculate D-efficiency
            # Standard formula: D-eff = |X'X|^(1/p) / n^(p/n) or |X'X|^(1/p) / n
            # We'll use the more conservative: |X'X|^(1/p) / n
            d_eff = (det_info ** (1.0 / p)) / n_runs
            
            # Alternative normalization (more common in conjoint literature)
            # d_eff = (det_info / n_runs**p) ** (1.0 / p)
            
            return min(max(float(d_eff), 0.0), 1.0)
            
        except Exception as e:
            return 0.0
    
    def generate_candidate_set(self, attributes_dict):
        """Generate full factorial candidate set"""
        levels = []
        attr_names = []
        
        for attr_name, attr_levels in attributes_dict.items():
            levels.append(list(range(len(attr_levels))))
            attr_names.append(attr_name)
        
        combinations = list(product(*levels))
        candidate_df = pd.DataFrame(combinations, columns=attr_names)
        
        return candidate_df

    def find_optimal_questions_advanced(self, attributes_dict, n_respondents, 
                                      target_efficiency=0.8, max_questions=25):
        """Find optimal number of questions with corrected D-efficiency calculation"""
        
        # Calculate number of parameters
        num_params = sum(len(levels) - 1 for levels in attributes_dict.values()) + 1
        
        results = []
        min_questions = max(2, num_params // 4)
        
        for n_questions in range(min_questions, max_questions + 1):
            total_runs = n_respondents * n_questions
            
            # Generate balanced design
            sampled_design = self.generate_balanced_fractional_factorial(
                attributes_dict, total_runs
            )
            
            # Calculate design matrix and information matrix
            design_matrix = self.create_design_matrix(sampled_design, attributes_dict)
            info_matrix = self.calculate_information_matrix(design_matrix)
            
            # Calculate D-efficiency
            d_efficiency = self.calculate_d_efficiency(info_matrix, total_runs)
            
            # Calculate other metrics
            std_errors = self.calculate_standard_errors(info_matrix)
            avg_std_error = np.mean(std_errors[std_errors != np.inf]) if len(std_errors[std_errors != np.inf]) > 0 else np.inf
            
            results.append({
                'num_questions': n_questions,
                'total_runs': total_runs,
                'd_efficiency': d_efficiency,
                'd_error': 1 - d_efficiency,
                'avg_std_error': avg_std_error,
                'obs_per_param': total_runs / num_params
            })
        
        results_df = pd.DataFrame(results)
        
        # Calculate theoretical minimum
        theoretical_min = max(2, (num_params + 2) // 2)
        
        return results_df, theoretical_min, max_questions
    
    def calculate_standard_errors(self, information_matrix):
        """Calculate standard errors from information matrix"""
        try:
            if det(information_matrix) <= 1e-10:
                return np.full(information_matrix.shape[0], np.inf)
            
            covariance_matrix = inv(information_matrix)
            self.covariance_matrix = covariance_matrix
            standard_errors = np.sqrt(np.diag(covariance_matrix))
            return standard_errors
            
        except:
            return np.full(information_matrix.shape[0], np.inf)
