"""
Enhanced Conjoint Analysis Utils
Advanced statistical methods for conjoint design optimization
"""

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import chi2
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class AdvancedConjointAnalyzer:
    """Advanced statistical methods for conjoint analysis"""
    
    def __init__(self):
        self.information_matrix = None
        self.covariance_matrix = None
        
    def create_design_matrix(self, profiles_df, attributes_dict):
        """
        Create design matrix with proper dummy coding
        """
        design_matrix = pd.DataFrame()
        
        for attr_name, levels in attributes_dict.items():
            if attr_name in profiles_df.columns:
                # Create dummy variables (effects coding)
                for i, level in enumerate(levels[:-1]):  # Drop last level as reference
                    col_name = f"{attr_name}_{level}"
                    design_matrix[col_name] = (profiles_df[attr_name] == i).astype(int)
        
        return design_matrix
    
    def calculate_information_matrix(self, design_matrix):
        """Calculate information matrix (X'X)"""
        X = design_matrix.values
        self.information_matrix = np.dot(X.T, X)
        return self.information_matrix
    
    def calculate_d_efficiency(self, information_matrix, n_runs):
        """
        Calculate D-efficiency with proper statistical formulation
        """
        try:
            if information_matrix.shape[0] == 0:
                return 0.0
            
            det_info = np.linalg.det(information_matrix)
            if det_info <= 0:
                return 0.0
            
            p = information_matrix.shape[0]  # Number of parameters
            d_efficiency = (det_info / n_runs) ** (1/p)
            
            return min(d_efficiency, 1.0)
        except:
            return 0.0
    
    def calculate_a_efficiency(self, information_matrix):
        """Calculate A-efficiency (trace criterion)"""
        try:
            if information_matrix.shape[0] == 0:
                return 0.0
            
            inv_info = np.linalg.inv(information_matrix)
            trace_inv = np.trace(inv_info)
            p = information_matrix.shape[0]
            
            a_efficiency = p / trace_inv
            return a_efficiency
        except:
            return 0.0
    
    def calculate_g_efficiency(self, information_matrix, candidate_points):
        """Calculate G-efficiency (minimax criterion)"""
        try:
            if information_matrix.shape[0] == 0:
                return 0.0
            
            inv_info = np.linalg.inv(information_matrix)
            max_variance = 0
            
            for point in candidate_points:
                x = np.array(point).reshape(-1, 1)
                variance = np.dot(np.dot(x.T, inv_info), x)[0, 0]
                max_variance = max(max_variance, variance)
            
            p = information_matrix.shape[0]
            g_efficiency = p / max_variance if max_variance > 0 else 0.0
            
            return g_efficiency
        except:
            return 0.0
    
    def calculate_standard_errors(self, information_matrix):
        """Calculate standard errors of parameter estimates"""
        try:
            if information_matrix.shape[0] == 0:
                return np.array([])
            
            inv_info = np.linalg.inv(information_matrix)
            standard_errors = np.sqrt(np.diag(inv_info))
            
            return standard_errors
        except:
            return np.array([])
    
    def calculate_correlation_matrix(self, information_matrix):
        """Calculate correlation matrix of parameter estimates"""
        try:
            if information_matrix.shape[0] == 0:
                return np.array([[]])
            
            inv_info = np.linalg.inv(information_matrix)
            std_errors = np.sqrt(np.diag(inv_info))
            
            # Calculate correlation matrix
            correlation_matrix = np.zeros_like(inv_info)
            for i in range(inv_info.shape[0]):
                for j in range(inv_info.shape[1]):
                    if std_errors[i] > 0 and std_errors[j] > 0:
                        correlation_matrix[i, j] = inv_info[i, j] / (std_errors[i] * std_errors[j])
            
            return correlation_matrix
        except:
            return np.array([[]])
    
    def calculate_power_analysis(self, information_matrix, effect_size, alpha=0.05):
        """
        Calculate statistical power for detecting effects
        """
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
                if se > 0:
                    t_stat = effect / se
                    power = 1 - chi2.cdf(t_critical**2, df=1, loc=t_stat**2)
                    power_values.append(power)
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
    
    def optimize_design_advanced(self, attributes_dict, n_respondents, 
                                optimization_criterion='D', max_iterations=1000):
        """
        Advanced design optimization using coordinate exchange algorithm
        """
        # Generate candidate set
        candidate_profiles = self.generate_candidate_set(attributes_dict)
        
        # Initialize with random design
        n_runs = min(len(candidate_profiles), n_respondents * 8)  # Initial guess
        current_design = candidate_profiles.sample(n=n_runs, random_state=42)
        
        best_efficiency = 0
        best_design = current_design.copy()
        
        for iteration in range(max_iterations):
            improved = False
            
            for run_idx in range(len(current_design)):
                current_efficiency = self.evaluate_design(current_design, optimization_criterion)
                
                # Try replacing current run with each candidate
                for candidate_idx in range(len(candidate_profiles)):
                    test_design = current_design.copy()
                    test_design.iloc[run_idx] = candidate_profiles.iloc[candidate_idx]
                    
                    test_efficiency = self.evaluate_design(test_design, optimization_criterion)
                    
                    if test_efficiency > current_efficiency:
                        current_design = test_design
                        improved = True
                        break
                
                if improved:
                    break
            
            current_efficiency = self.evaluate_design(current_design, optimization_criterion)
            if current_efficiency > best_efficiency:
                best_efficiency = current_efficiency
                best_design = current_design.copy()
            
            if not improved:
                break
        
        return best_design, best_efficiency
    
    def generate_candidate_set(self, attributes_dict):
        """Generate full candidate set of profiles"""
        from itertools import product
        
        levels = [list(range(len(levels))) for levels in attributes_dict.values()]
        combinations = list(product(*levels))
        
        candidate_df = pd.DataFrame(combinations, columns=list(attributes_dict.keys()))
        return candidate_df
    
    def evaluate_design(self, design_df, criterion='D'):
        """Evaluate design using specified criterion"""
        design_matrix = pd.get_dummies(design_df, drop_first=True)
        information_matrix = self.calculate_information_matrix(design_matrix)
        
        if criterion == 'D':
            return self.calculate_d_efficiency(information_matrix, len(design_df))
        elif criterion == 'A':
            return self.calculate_a_efficiency(information_matrix)
        elif criterion == 'G':
            candidate_points = self.generate_candidate_set({col: [0, 1] for col in design_matrix.columns})
            return self.calculate_g_efficiency(information_matrix, candidate_points.values)
        else:
            return self.calculate_d_efficiency(information_matrix, len(design_df))
    
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
            'D_efficiency': self.calculate_d_efficiency(information_matrix, len(design_df)),
            'A_efficiency': self.calculate_a_efficiency(information_matrix),
            'standard_errors': self.calculate_standard_errors(information_matrix),
            'correlation_matrix': self.calculate_correlation_matrix(information_matrix),
            'condition_number': np.linalg.cond(information_matrix) if information_matrix.shape[0] > 0 else np.inf
        }
        
        # Power analysis
        power_results = self.calculate_power_analysis(information_matrix, effect_size=0.5)
        metrics.update(power_results)
        
        return metrics

def validate_design_balance(design_df, attributes_dict):
    """Check if design is balanced across attribute levels"""
    balance_report = {}
    
    for attr_name, levels in attributes_dict.items():
        if attr_name in design_df.columns:
            value_counts = design_df[attr_name].value_counts().sort_index()
            balance_report[attr_name] = {
                'counts': value_counts.to_dict(),
                'is_balanced': value_counts.std() / value_counts.mean() < 0.1 if value_counts.mean() > 0 else False,
                'balance_ratio': value_counts.min() / value_counts.max() if value_counts.max() > 0 else 0
            }
    
    return balance_report

def calculate_sample_size_requirements(attributes_dict, desired_power=0.8, effect_size=0.5, alpha=0.05):
    """Calculate minimum sample size requirements"""
    
    # Calculate degrees of freedom
    df_total = sum(len(levels) - 1 for levels in attributes_dict.values())
    
    # Approximate sample size calculation
    # This is a simplified version - in practice, you'd use power analysis
    from scipy.stats import norm
    
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(desired_power)
    
    # Cohen's rule of thumb for multiple regression
    min_sample_size = max(50 + 8 * df_total, (z_alpha + z_beta)**2 / (effect_size**2))
    
    return {
        'min_sample_size': int(min_sample_size),
        'degrees_of_freedom': df_total,
        'recommended_oversample': int(min_sample_size * 1.2),
        'parameters': {
            'desired_power': desired_power,
            'effect_size': effect_size,
            'alpha': alpha
        }
    }