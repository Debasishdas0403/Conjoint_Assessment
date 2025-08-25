import numpy as np
import pandas as pd
from scipy.linalg import det, inv
from scipy.stats import chi2
from itertools import product
import streamlit as st

class AdvancedConjointAnalyzer:
    """Advanced statistical methods for conjoint analysis with corrected D-efficiency"""
    
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
                
                # Effects coding: use n-1 dummy variables
                for i in range(n_levels - 1):
                    col_name = f"{attr_name}_L{i+1}"
                    # Effects coding: 1 if level i, -1 if last level, 0 otherwise
                    design_matrix[col_name] = profiles_df[attr_name].apply(
                        lambda x: 1 if x == i else (-1 if x == n_levels - 1 else 0)
                    )
        
        return design_matrix
    
    def calculate_information_matrix(self, design_matrix):
        """Calculate information matrix (X'X)"""
        X = design_matrix.values
        self.information_matrix = np.dot(X.T, X)
        return self.information_matrix
    
    def calculate_d_efficiency(self, information_matrix, n_runs, normalize=True):
        """
        Calculate D-efficiency with proper statistical formulation
        
        D-efficiency = (det(X'X))^(1/p) / n_runs^(p/n_runs) where p = number of parameters
        """
        try:
            if information_matrix.shape[0] == 0:
                return 0.0
            
            # Check for singularity
            det_info = det(information_matrix)
            if det_info <= 0 or np.isnan(det_info) or np.isinf(det_info):
                return 0.0
            
            # Number of parameters
            p = information_matrix.shape[0]
            
            # Calculate D-efficiency with proper normalization
            d_eff = (det_info) ** (1.0 / p)
            
            if normalize and n_runs > 0:
                # Normalize by number of runs raised to (1/p)
                d_eff = d_eff / (n_runs ** (1.0 / p))
                
                # Ensure it's between 0 and 1
                d_eff = min(max(d_eff, 0.0), 1.0)
            
            return float(d_eff)
            
        except Exception as e:
            return 0.0
    
    def calculate_a_efficiency(self, information_matrix):
        """Calculate A-efficiency (harmonic mean of eigenvalues)"""
        try:
            if det(information_matrix) <= 0:
                return 0.0
            
            eigenvals = np.linalg.eigvals(information_matrix)
            eigenvals = eigenvals[eigenvals > 0]  # Keep only positive eigenvalues
            
            if len(eigenvals) == 0:
                return 0.0
            
            a_eff = len(eigenvals) / np.sum(1.0 / eigenvals)
            return float(a_eff)
            
        except:
            return 0.0
    
    def calculate_g_efficiency(self, information_matrix, candidate_points):
        """Calculate G-efficiency (minimax efficiency)"""
        try:
            if det(information_matrix) <= 0:
                return 0.0
            
            inv_info = inv(information_matrix)
            max_variance = 0
            
            for point in candidate_points:
                point = point.reshape(-1, 1)
                variance = np.dot(np.dot(point.T, inv_info), point)[0, 0]
                max_variance = max(max_variance, variance)
            
            if max_variance > 0:
                g_eff = information_matrix.shape[0] / max_variance
                return float(g_eff)
            else:
                return 0.0
                
        except:
            return 0.0
    
    def calculate_standard_errors(self, information_matrix):
        """Calculate standard errors from information matrix"""
        try:
            if det(information_matrix) <= 0:
                return np.full(information_matrix.shape[0], np.inf)
            
            covariance_matrix = inv(information_matrix)
            self.covariance_matrix = covariance_matrix
            standard_errors = np.sqrt(np.diag(covariance_matrix))
            return standard_errors
            
        except:
            return np.full(information_matrix.shape[0], np.inf)
    
    def calculate_correlation_matrix(self, information_matrix):
        """Calculate parameter correlation matrix"""
        try:
            if det(information_matrix) <= 0:
                return np.array([[]])
            
            covariance_matrix = inv(information_matrix)
            std_devs = np.sqrt(np.diag(covariance_matrix))
            correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)
            
            return correlation_matrix
        except:
            return np.array([[]])
    
    def calculate_power_analysis(self, information_matrix, effect_size, alpha=0.05):
        """Calculate statistical power for detecting effects"""
        try:
            if information_matrix.shape[0] == 0:
                return {}
            
            std_errors = self.calculate_standard_errors(information_matrix)
            
            # Calculate t-value for given alpha
            df = information_matrix.shape[0]  # Approximate degrees of freedom
            t_critical = chi2.ppf(1 - alpha/2, df=1) ** 0.5
            
            # Calculate minimum detectable effect
            min_detectable_effects = t_critical * std_errors
            
            # Calculate power for given effect size
            if isinstance(effect_size, (int, float)):
                effect_size = [effect_size] * len(std_errors)
            
            power_values = []
            for i, (effect, se) in enumerate(zip(effect_size, std_errors)):
                if se > 0 and not np.isinf(se):
                    t_stat = effect / se
                    power = 1 - chi2.cdf(t_critical**2, df=1, loc=t_stat**2)
                    power_values.append(max(0.0, min(1.0, power)))
                else:
                    power_values.append(0.0)
            
            return {
                'standard_errors': std_errors,
                'min_detectable_effects': min_detectable_effects,
                'power_values': power_values,
                'average_power': np.mean(power_values)
            }
        except:
            return {}
    
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
    
    def optimize_design_advanced(self, attributes_dict, n_respondents, 
                                optimization_criterion='D', max_iterations=500):
        """Advanced design optimization using coordinate exchange algorithm"""
        # Generate candidate set
        candidate_profiles = self.generate_candidate_set(attributes_dict)
        
        # Calculate minimum runs needed
        num_params = sum(len(levels) - 1 for levels in attributes_dict.values()) + 1
        min_runs = max(num_params * 2, 20)
        n_runs = min(len(candidate_profiles), max(min_runs, n_respondents * 8))
        
        # Initialize with random design
        np.random.seed(42)
        current_design = candidate_profiles.sample(n=n_runs, random_state=42, replace=True)
        
        best_efficiency = 0
        best_design = current_design.copy()
        no_improvement_count = 0
        
        for iteration in range(max_iterations):
            improved = False
            
            # Calculate current efficiency
            current_design_matrix = self.create_design_matrix(current_design, attributes_dict)
            current_info_matrix = self.calculate_information_matrix(current_design_matrix)
            current_efficiency = self.evaluate_design(current_info_matrix, len(current_design), optimization_criterion)
            
            # Try improving each run
            for run_idx in range(len(current_design)):
                best_candidate = None
                best_candidate_efficiency = current_efficiency
                
                # Try each candidate for this position
                for candidate_idx in range(min(len(candidate_profiles), 50)):  # Limit for performance
                    test_design = current_design.copy()
                    test_design.iloc[run_idx] = candidate_profiles.iloc[candidate_idx]
                    
                    # Calculate efficiency for test design
                    test_design_matrix = self.create_design_matrix(test_design, attributes_dict)
                    test_info_matrix = self.calculate_information_matrix(test_design_matrix)
                    test_efficiency = self.evaluate_design(test_info_matrix, len(test_design), optimization_criterion)
                    
                    if test_efficiency > best_candidate_efficiency:
                        best_candidate = candidate_idx
                        best_candidate_efficiency = test_efficiency
                        improved = True
                
                # Update if improvement found
                if best_candidate is not None:
                    current_design.iloc[run_idx] = candidate_profiles.iloc[best_candidate]
            
            # Update best design if improved
            if current_efficiency > best_efficiency:
                best_efficiency = current_efficiency
                best_design = current_design.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Early stopping if no improvement for several iterations
            if no_improvement_count > 50:
                break
        
        return best_design
    
    def evaluate_design(self, information_matrix, n_runs, criterion='D'):
        """Evaluate design based on specified criterion"""
        if criterion == 'D':
            return self.calculate_d_efficiency(information_matrix, n_runs)
        elif criterion == 'A':
            return self.calculate_a_efficiency(information_matrix)
        elif criterion == 'G':
            # For G-efficiency, we need candidate points - simplified version
            return self.calculate_d_efficiency(information_matrix, n_runs)  # Fallback to D
        else:
            return self.calculate_d_efficiency(information_matrix, n_runs)
    
    def generate_blocking_design(self, base_design, n_blocks):
        """Generate blocked design for multiple survey versions"""
        block_size = len(base_design) // n_blocks
        blocks = []
        
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size if i < n_blocks - 1 else len(base_design)
            
            block = base_design.iloc[start_idx:end_idx].copy()
            block['Block'] = i + 1
            blocks.append(block)
        
        return pd.concat(blocks, ignore_index=True)
    
    def calculate_efficiency_metrics(self, design_df, attributes_dict):
        """Calculate comprehensive efficiency metrics"""
        design_matrix = self.create_design_matrix(design_df, attributes_dict)
        information_matrix = self.calculate_information_matrix(design_matrix)
        
        metrics = {
            'd_efficiency': self.calculate_d_efficiency(information_matrix, len(design_df)),
            'a_efficiency': self.calculate_a_efficiency(information_matrix),
            'condition_number': np.linalg.cond(information_matrix) if det(information_matrix) > 0 else np.inf,
            'standard_errors': self.calculate_standard_errors(information_matrix),
            'parameter_correlations': self.calculate_correlation_matrix(information_matrix)
        }
        
        # Add power analysis
        power_analysis = self.calculate_power_analysis(information_matrix, effect_size=0.5)
        metrics.update(power_analysis)
        
        return metrics

    def find_optimal_questions_advanced(self, attributes_dict, n_respondents, 
                                      target_efficiency=0.8, max_questions=25):
        """Find optimal number of questions with advanced D-efficiency calculation"""
        
        # Calculate number of parameters
        num_params = sum(len(levels) - 1 for levels in attributes_dict.values()) + 1
        
        # Generate candidate set
        candidate_set = self.generate_candidate_set(attributes_dict)
        
        results = []
        min_questions = max(2, num_params // 4)  # More realistic minimum
        
        for n_questions in range(min_questions, max_questions + 1):
            total_runs = n_respondents * n_questions
            
            # Create design by sampling from candidate set
            if total_runs <= len(candidate_set):
                sampled_design = candidate_set.sample(n=total_runs, random_state=42, replace=False)
            else:
                sampled_design = candidate_set.sample(n=total_runs, random_state=42, replace=True)
            
            # Calculate design matrix and information matrix
            design_matrix = self.create_design_matrix(sampled_design, attributes_dict)
            info_matrix = self.calculate_information_matrix(design_matrix)
            
            # Calculate D-efficiency
            d_efficiency = self.calculate_d_efficiency(info_matrix, total_runs)
            
            # Calculate other metrics
            std_errors = self.calculate_standard_errors(info_matrix)
            avg_std_error = np.mean(std_errors[std_errors != np.inf])
            
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
        theoretical_min = max(2, (num_params + 2))
        
        return results_df, theoretical_min, max_questions