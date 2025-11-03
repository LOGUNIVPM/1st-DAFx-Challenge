class ParamRange:
    def __init__(self, low, high):
        self.low = low
        self.high = high
    
    def is_fixed(self):
        """Check if this parameter is fixed (low == high)."""
        return self.low == self.high
    
    def get_value(self):
        """Get the fixed value (only valid if is_fixed() returns True)."""
        if self.is_fixed():
            return self.low
        else:
            raise ValueError("Parameter is not fixed, cannot get single value")


def get_variable_params():
    """
    Get only the parameters that are variable (not fixed).
    
    Returns:
        dict: Dictionary containing only variable parameters
    """
    return {k: v for k, v in params.items() if not v.is_fixed()}


def get_fixed_params():
    """
    Get only the parameters that are fixed.
    
    Returns:
        dict: Dictionary containing only fixed parameters with their values
    """
    return {k: v.get_value() for k, v in params.items() if v.is_fixed()}


def variable_params_to_full_params(variable_values):
    """
    Convert variable parameter values to a full parameter dictionary.
    
    Args:
        variable_values: List or array of values for variable parameters only
        
    Returns:
        dict: Full parameter dictionary with fixed and variable values
    """
    variable_param_keys = list(get_variable_params().keys())
    
    if len(variable_values) != len(variable_param_keys):
        raise ValueError(f"Expected {len(variable_param_keys)} variable values, got {len(variable_values)}")
    
    # Start with fixed parameters
    full_params = get_fixed_params()
    
    # Add variable parameters
    for i, key in enumerate(variable_param_keys):
        full_params[key] = variable_values[i]
    
    return full_params


def full_params_to_variable_params(full_params):
    """
    Extract only the variable parameter values from a full parameter dictionary.
    
    Args:
        full_params: Dictionary with all parameters
        
    Returns:
        list: Values for variable parameters only, in consistent order
    """
    variable_param_keys = list(get_variable_params().keys())
    return [full_params[key] for key in variable_param_keys]

# Revised parameter ranges: removes ambiguities (after discussion on Oct. 7th)
params = {
    'Lx': ParamRange(1.0, 1.0), # Fixed
    'Ly': ParamRange(1.1, 4.0),  
    'h': ParamRange(0.001, 0.005),
    'T0': ParamRange(0.01, 1000.0),
    'rho': ParamRange(2430.0, 21230.0),
    'E': ParamRange(6.7e10, 22.0e10),
    'nu': ParamRange(0.25, 0.25), # Fixed
    'T60_DC': ParamRange(6.0, 6.0), # Fixed
    'T60_F1': ParamRange(2.0, 2.0), # Fixed
    'loss_F1': ParamRange(500.0, 500.0), # Fixed
    'fp_x': ParamRange(0.335, 0.335), # Fixed
    'fp_y': ParamRange(0.467, 0.467), # Fixed
    'op_x': ParamRange(0.51, 1.0),
    'op_y': ParamRange(0.51, 1.0)
}