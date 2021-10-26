import numpy as np
from scipy.integrate import quad

def Einstein(T, T_e, atoms):
    '''
    This is a function which implements the Einstein model for heat capacity.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_e (float) = characterstic energy expressed as temperature. 
    T_e = w_e/k_B where k_B is the Boltzmann constant
    atoms (float) = atoms per formula unit
    
    Returns
    -------
    (float) The heat capacity at the temperature T with units joule per kelvin (J/K)
    '''
    R = 8.3144 # This is the ideal gas constant
    
    # The left-most term in the product for C_e above
    A = 3*R*(T_e/T)**2 
    
    # The right-most term in the product for C_e above
    B = (np.exp(-T_e/T))/(1-np.exp(-T_e/T))**2 
    
    return atoms*A*B

def Einstein_graphing(T, T_e, atoms):
    '''
    This is a function which grpahs the Einstein model for heat capacity.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_e (float) = characterstic energy expressed as temperature. 
    T_e = w_e/k_B where k_B is the Boltzmann constant
    atoms (float) = atoms per formula unit
    
    Returns
    -------
    An array of heat capacities over the interval of temperatures provided at the characteristic temp T_e
    '''
    return [Einstein(x, T_e, atoms) for x in T]

def Debye(T, T_d, atoms):
    '''
    This is a function which implements the Debye model for heat capacity.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_d (float) = characterstic energy expressed as temperature. 
    T_d = w_d/k_B where k_B is the Boltzmann constant
    atoms (float) = atoms per formula unit
    
    Returns
    -------
    (float) The heat capacity at the temperature T with units joule per kelvin (J/K)
    '''
    R = 8.3144 # This is the ideal gas constant
    
    # Everything that doesn't involve an integral in C_d above
    A = 9*R*(T/T_d)**3 
    
    # Everything that does involve an integral in C_d above. 
    # Note that quad returns a tuple: (est. value of integral, upper bound on the error)
    integral = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d/T)[0]
    
    return atoms*A*integral

def Debye_graphing(T, T_d, atoms):
    '''
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_d (float) = characterstic energy expressed as temperature. 
    T_d = w_d/k_B where k_B is the Boltzmann constant
    atoms (float) = atoms per formula unit
    
    Returns
    -------
    An array of heat capacities over the interval of temperatures provided at the characteristic temp T_d
    '''
    return [Debye(x, T_d, atoms) for x in T]

def Einstein_alt(T, T_e, atoms):
    '''
    This is a function which implements the Einstein model for heat capacity.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_e (float) = characterstic energy expressed as temperature. 
    T_e = w_e/k_B where k_B is the Boltzmann constant
    atoms (float) = atoms per formula unit
    
    Returns
    -------
    (float) The heat capacity at the temperature T with units joule per kelvin (J/K)
    '''
    R = 8.3144 # This is the ideal gas constant
    
    # The left-most term in the product for C_e above
    A = 3*R*T_e**2/(T**5) 
    
    # The right-most term in the product for C_e above
    B = (np.exp(-T_e/T))/(1-np.exp(-T_e/T))**2 
    
    return atoms*A*B

def Einstein_alt_graphing(T, T_e, atoms):
    '''
    This is a function which graphs the Einstein model for heat capacity.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_e (float) = characterstic energy expressed as temperature. 
    T_e = w_e/k_B where k_B is the Boltzmann constant
    atoms (float) = atoms per formula unit
    
    Returns
    -------
    (array) The heat capacities at the range of temperatures T with units joule per kelvin (J/K)
    '''
    return [Einstein_alt(x, T_e, atoms) for x in T]

def Debye_alt(T, T_d, atoms):
    '''
    This is a function which implements the Debye model for heat capacity.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_d (float) = characterstic energy expressed as temperature. 
    T_d = w_d/k_B where k_B is the Boltzmann constant
    atoms (float) = atoms per formula unit
    
    Returns
    -------
    (float) The heat capacity at the temperature T with units joule per kelvin (J/K)
    '''
    R = 8.3144 # This is the ideal gas constant
    
    # Everything that doesn't involve an integral in C_d above
    A = 9*R*(1/T_d)**3 
    
    # Everything that does involve an integral in C_d above. 
    # Note that quad returns a tuple: (est. value of integral, upper bound on the error)
    integral = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d/T)[0]
    
    return atoms*A*integral

def Debye_graphing_alt(T, T_d, atoms):
    '''
    This is a function which graphs the Debye model for heat capacity.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_d (float) = characterstic energy expressed as temperature. 
    T_d = w_d/k_B where k_B is the Boltzmann constant
    atoms (float) = atoms per formula unit
    
    Returns
    -------
    (array) The heat capacities at the range of temperature T with units joule per kelvin (J/K)
    '''
    return [Debye_alt(x, T_d, atoms) for x in T]

def Debye_2(T, T_d, T_d2, atoms, a1, a2):
    '''
    This is a function which implements the Debye model for heat capacity using a linear combinations of two curves.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_d (float) = characterstic energy expressed as temperature. 
    T_d = w_d/k_B where k_B is the Boltzmann constant
    T_d2 (float) = character characterstic energy expressed as temperature for the second curve
    atoms (float) = atoms per formula unit
    a1 (float) = coefficient of the first term in the sum
    a2 (float) = coefficient of the second term in the sum
    
    Returns
    -------
    (float) The heat capacity at the temperature T with units joule per kelvin (J/K)
    '''
    R = 8.3144 # This is the ideal gas constant
    
    # Everything that doesn't involve an integral in C_d above
    A = 9*R*(T/T_d)**3 
    A2 = 9*R*(T/T_d2)**3 
    
    # Everything that does involve an integral in C_d above. 
    # Note that quad returns a tuple: (est. value of integral, upper bound on the error)
    integral = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d/T)[0]
    integral2 = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d2/T)[0]
    
    return a1*atoms*A*integral + a2*atoms*A2*integral2

def Debye_graphing_2(T, T_d, T_d2, atoms, a1, a2):
    '''
    Parameters
    ----------
    T (array) = temperature to evaluate the function at
    T_d (float) = characterstic energy expressed as temperature. 
    T_d = w_d/k_B where k_B is the Boltzmann constant
    atoms (float) = atoms per formula unit
    
    Returns
    -------
    An array of heat capacities over the interval of temperatures provided at the characteristic temp T_d
    '''
    return [Debye_2(x, T_d, T_d2, atoms, a1, a2) for x in T]

def Debye_2_alt(T, T_d, T_d2, atoms, a1, a2):
    '''
    This is a function which implements the Debye model for 
    heat capacity using a linear combinations of two curves.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    (float) The heat capacity at the temperature T with units joule per kelvin (J/K)
    '''
    R = 8.3144 # This is the ideal gas constant
    
    # Everything that doesn't involve an integral in C_d above
    A = 9*R*(1/T_d)**3 
    A2 = 9*R*(1/T_d2)**3 
    
    # Everything that does involve an integral in C_d above. 
    # Note that quad returns a tuple: (est. value of integral, upper bound on the error)
    integral = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d/T)[0]
    integral2 = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d2/T)[0]
    
    return a1*atoms*A*integral + a2*atoms*A2*integral2

def Debye_graphing_2_alt(T, T_d, T_d2, atoms, a1, a2):
    '''
    Parameters
    ----------
    T (array) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    An array of heat capacities over the interval of temperatures 
    provided at the characteristic temp T_d
    '''
    return [Debye_2_alt(x, T_d, T_d2, atoms, a1, a2) for x in T]

def Debye_3(T, T_d, T_d2, T_d3, atoms, a1, a2, a3):
    '''
    This is a function which implements the Debye model for 
    heat capacity using a linear combinations of two curves.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    (float) The heat capacity at the temperature T with units joule per kelvin (J/K)
    '''
    R = 8.3144 # This is the ideal gas constant
    
    # Everything that doesn't involve an integral in C_d above
    A = 9*R*(T/T_d)**3 
    A2 = 9*R*(T/T_d2)**3 
    A3 = 9*R*(T/T_d3)**3 
    
    # Everything that does involve an integral in C_d above. 
    # Note that quad returns a tuple: (est. value of integral, upper bound on the error)
    integral = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d/T)[0]
    integral2 = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d2/T)[0]
    integral3 = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d3/T)[0]
    
    return a1*atoms*A*integral + a2*atoms*A2*integral2 + a3*atoms*A3*integral3

def Debye_graphing_3(T, T_d, T_d2, T_d3, atoms, a1, a2, a3):
    '''
    Parameters
    ----------
    T (array) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    An array of heat capacities over the interval of temperatures 
    provided at the characteristic temp T_d
    '''
    return [Debye_3(x, T_d, T_d2, T_d3, atoms, a1, a2, a3) for x in T]

def Debye_3_alt(T, T_d, T_d2, Td_3, atoms, a1, a2, a3):
    '''
    This is a function which implements the Debye model for 
    heat capacity using a linear combinations of two curves.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    (float) The heat capacity at the temperature T with units joule per kelvin (J/K)
    '''
    R = 8.3144 # This is the ideal gas constant
    
    # Everything that doesn't involve an integral in C_d above
    A = 9*R*(1/T_d)**3 
    A2 = 9*R*(1/T_d2)**3 
    A3 = 9*R*(1/T_d2)**3
    
    # Everything that does involve an integral in C_d above. 
    # Note that quad returns a tuple: (est. value of integral, upper bound on the error)
    integral = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d/T)[0]
    integral2 = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d2/T)[0]
    integral3 = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d2/T)[0]
    
    return a1*atoms*A*integral + a2*atoms*A2*integral2 + a3*atoms*A3*integral3

def Debye_graphing_3_alt(T, T_d, T_d2, T_d3, atoms, a1, a2, a3):
    '''
    Parameters
    ----------
    T (array) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    An array of heat capacities over the interval of temperatures 
    provided at the characteristic temp T_d
    '''
    return [Debye_3_alt(x, T_d, T_d2, T_d3, atoms, a1, a2, a3) for x in T]

def Einstein_2(T, T_e, T_e2, atoms, a1, a2):
    '''
    This is a function which implements the Debye model for 
    heat capacity using a linear combinations of two curves.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    (float) The heat capacity at the temperature T with units joule per kelvin (J/K)
    '''
    R = 8.3144 # This is the ideal gas constant
    
    # The left-most term in the product for C_e above
    A1 = 3*R*(T_e/T)**2 
    A2 = 3*R*(T_e2/T)**2 
    
    # The right-most term in the product for C_e above
    B1 = (np.exp(-T_e/T))/(1-np.exp(-T_e/T))**2
    B2 = (np.exp(-T_e2/T))/(1-np.exp(-T_e2/T))**2 
    
    return a1*atoms*A1*B1 + a2*atoms*A2*B2

def Einstein_graphing_2(T, T_e, T_e2, atoms, a1, a2):
    '''
    Parameters
    ----------
    T (array) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    An array of heat capacities over the interval of temperatures 
    provided at the characteristic temp T_d
    '''
    return [Einstein_2(x, T_e, T_e2, atoms, a1, a2) for x in T]

def Einstein_2_alt(T, T_e, T_e2, atoms, a1, a2):
    '''
    This is a function which implements the Debye model for 
    heat capacity using a linear combinations of two curves.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    (float) The heat capacity at the temperature T with units joule per kelvin (J/K)
    '''
    R = 8.3144 # This is the ideal gas constant
    
    # The left-most term in the product for C_e above
    A1 = 3*R*T_e**2/(T**5) 
    A2 = 3*R*T_e2**2/(T**5)
    
    # The right-most term in the product for C_e above
    B1 = (np.exp(-T_e/T))/(1-np.exp(-T_e/T))**2
    B2 = (np.exp(-T_e2/T))/(1-np.exp(-T_e2/T))**2
    
    return a1*atoms*A1*B1 + a2*atoms*A2*B2

def Einstein_graphing_2_alt(T, T_e, T_e2, atoms, a1, a2):
    '''
    Parameters
    ----------
    T (array) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    An array of heat capacities over the interval of temperatures 
    provided at the characteristic temp T_d
    '''
    return [Einstein_2_alt(x, T_e, T_e2, atoms, a1, a2) for x in T]

def Model_2(T, T_d, T_e, atoms, a1, a2):
    '''
    This is a function which implements the Debye model for 
    heat capacity using a linear combinations of two curves.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    (float) The heat capacity at the temperature T with units joule per kelvin (J/K)
    '''
    R = 8.3144 # This is the ideal gas constant
    
    # Everything that doesn't involve an integral in C_d above
    AD = 9*R*(T/T_d)**3 
    AE = 3*R*(T_e/T)**2 
    
    # Everything that does involve an integral in C_d above. 
    # Note that quad returns a tuple: (est. value of integral, upper bound on the error)
    BD = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d/T)[0]
    BE = (np.exp(-T_e/T))/(1-np.exp(-T_e/T))**2
    
    return a1*atoms*AD*BD + a2*atoms*AE*BE

def Model_graph_2(T, T_d, T_e, atoms, a1, a2):
    '''
    Parameters
    ----------
    T (array) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    An array of heat capacities over the interval of temperatures 
    provided at the characteristic temp T_d
    '''
    return [Model_2(x, T_d, T_e, atoms, a1, a2) for x in T]

def Mult_2_alt(T, T_d, T_e, atoms, a1, a2):
    '''
    This is a function which implements the Debye model for 
    heat capacity using a linear combinations of two curves.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    (float) The heat capacity at the temperature T with units joule per kelvin (J/K)
    '''
    R = 8.3144 # This is the ideal gas constant
    
    # Everything that doesn't involve an integral in C_d above
    AD = 9*R*(1/T_d)**3 
    AE = 3*R*T_e**2/(T**5) 
    
    # Everything that does involve an integral in C_d above. 
    # Note that quad returns a tuple: (est. value of integral, upper bound on the error)
    BD = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d/T)[0]
    BE = (np.exp(-T_e/T))/(1-np.exp(-T_e/T))**2
    
    return a1*atoms*AD*BD + a2*atoms*AE*BE
    
def Mult_graph_2_alt(T, T_d, T_e, atoms, a1, a2):
    '''
    Parameters
    ----------
    T (array) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    An array of heat capacities over the interval of temperatures 
    provided at the characteristic temp T_d
    '''
    return [Mult_2_alt(x, T_d, T_e, atoms, a1, a2) for x in T]

def Model_3E1(T, T_d, T_d2, T_e, atoms, a1, a2, a3):
    '''
    This is a function which implements the Debye model for 
    heat capacity using a linear combinations of two curves.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    (float) The heat capacity at the temperature T with units joule per kelvin (J/K)
    '''
    R = 8.3144 # This is the ideal gas constant
    
    # Everything that doesn't involve an integral in C_d above
    AD = 9*R*(T/T_d)**3 
    AD2 = 9*R*(T/T_d2)**3 
    AE = 3*R*(T_e/T)**2 
    
    # Everything that does involve an integral in C_d above. 
    # Note that quad returns a tuple: (est. value of integral, upper bound on the error)
    integralD = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d/T)[0]
    integralD2 = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d2/T)[0]
    BE = (np.exp(-T_e/T))/(1-np.exp(-T_e/T))**2
    
    return a1*atoms*AD*integralD + a2*atoms*AD2*integralD2 + a3*atoms*AE*BE

def Model_3E1_graphing(T, T_d, T_d2, T_e, atoms, a1, a2, a3):
    '''
    Parameters
    ----------
    T (array) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    An array of heat capacities over the interval of temperatures provided at the characteristic temp T_d
    '''
    return [Model_3E1(x, T_d, T_d2, T_e, atoms, a1, a2, a3) for x in T]

def Model_3E1_alt(T, T_d, T_d2, T_e, atoms, a1, a2, a3):
    '''
    This is a function which implements the Debye model for 
    heat capacity using a linear combinations of two curves.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    (float) The heat capacity at the temperature T with units joule per kelvin (J/K)
    '''
    R = 8.3144 # This is the ideal gas constant
    
    # Everything that doesn't involve an integral in C_d above
    AD = 9*R*(1/T_d)**3 
    AD2 = 9*R*(1/T_d2)**3 
    AE = 3*R*T_e**2/(T**5)
    
    # Everything that does involve an integral in C_d above. 
    # Note that quad returns a tuple: (est. value of integral, upper bound on the error)
    integralD = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d/T)[0]
    integralD2 = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d2/T)[0]
    BE = (np.exp(-T_e/T))/(1-np.exp(-T_e/T))**2
    
    return a1*atoms*AD*integralD + a2*atoms*AD2*integralD2 + a3*atoms*AE*BE

def Model_graphing_3E1_alt(T, T_d, T_d2, T_e, atoms, a1, a2, a3):
    '''
    Parameters
    ----------
    T (array) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    An array of heat capacities over the interval of temperatures 
    provided at the characteristic temp T_d
    '''
    return [Model_3E1_alt(x, T_d, T_d2, T_e, atoms, a1, a2, a3) for x in T]

def Model_3D1(T, T_d, T_e, T_e2, atoms, a1, a2, a3):
    '''
    This is a function which implements the Debye model for 
    heat capacity using a linear combinations of two curves.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    (float) The heat capacity at the temperature T with units joule per kelvin (J/K)
    '''
    R = 8.3144 # This is the ideal gas constant
    
    # Everything that doesn't involve an integral in C_d above
    AD = 9*R*(T/T_d)**3 
    AE = 3*R*(T_e/T)**2 
    AE2 = 3*R*(T_e2/T)**2 
    
    # Everything that does involve an integral in C_d above. 
    # Note that quad returns a tuple: (est. value of integral, upper bound on the error)
    integralD = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d/T)[0]
    BE = (np.exp(-T_e/T))/(1-np.exp(-T_e/T))**2
    BE2 = (np.exp(-T_e2/T))/(1-np.exp(-T_e2/T))**2
    
    return a1*atoms*AD*integralD + a2*atoms*AE*BE + a3*atoms*AE2*BE2

def Model_3D1_graphing(T, T_d, T_e, T_e2, atoms, a1, a2, a3):
    '''
    Parameters
    ----------
    T (array) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    An array of heat capacities over the interval of temperatures 
    provided at the characteristic temp T_d
    '''
    return [Model_3D1(x, T_d, T_e, T_e2, atoms, a1, a2, a3) for x in T]

def Model_3D1_alt(T, T_d, T_e, T_e2, atoms, a1, a2, a3):
    '''
    This is a function which implements the Debye model for 
    heat capacity using a linear combinations of two curves.
    Parameters
    ----------
    T (float) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    (float) The heat capacity at the temperature T with units joule per kelvin (J/K)
    '''
    R = 8.3144 # This is the ideal gas constant
    
    # Everything that doesn't involve an integral in C_d above
    AD = 9*R*(1/T_d)**3 
    AE = 3*R*T_e**2/(T**5) 
    AE2 = 3*R*T_e2**2/(T**5)
    
    # Everything that does involve an integral in C_d above. 
    # Note that quad returns a tuple: (est. value of integral, upper bound on the error)
    integralD = quad(lambda x: (x**4)*np.exp(-x)/(1-np.exp(-x))**2, 0, T_d/T)[0]
    BE = (np.exp(-T_e/T))/(1-np.exp(-T_e/T))**2
    BE2 = (np.exp(-T_e2/T))/(1-np.exp(-T_e2/T))**2
    
    return a1*atoms*AD*integralD + a2*atoms*AE*BE + a3*atoms*AE2*BE2

def Model_graphing_3D1_alt(T, T_d, T_e, T_e2, atoms, a1, a2, a3):
    '''
    Parameters
    ----------
    T (array) = temperature to evaluate the function at
    T_X (float) = Characteristic temperatures
    atoms (float) = number of atoms  per formula unit
    aX (float) = coefficients
    
    Returns
    -------
    An array of heat capacities over the interval of temperatures 
    provided at the characteristic temp T_d
    '''
    return [Model_3D1_alt(x, T_d, T_e, T_e2, atoms, a1, a2, a3) for x in T]

