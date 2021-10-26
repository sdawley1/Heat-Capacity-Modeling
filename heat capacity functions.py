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

plt.figure()

# Fitting the data and unpacking the fit parameters
pars2DA, cov2DA = curve_fit(
    Debye_graphing_2_alt, 
    temperatures, mod_heat_capacities, 
    bounds=(0,[np.inf, np.inf, 7, 7, 7])
)
Td_fit2DA, Td2_fit2DA, atoms_fit2DA, a1_fit2DA, a2_fit2DA = pars2DA 
        
# Plotting
plt.scatter(temperatures, mod_heat_capacities, s=10, c='r', label='Mystery Data')
plt.plot(temperatures, 
         Debye_graphing_2_alt(temperatures, 
                              Td_fit2DA, Td2_fit2DA, 
                              atoms_fit2DA, 
                              a1_fit2DA, a2_fit2DA), 
         alpha=1, lw=2, c='k', label='Best Fit')

plt.title('Fitting with a Sum of Two Debye Models Over Temperature Cubed')
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity over Temp Cubed')
plt.legend()

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

plt.figure()

pars, cov = curve_fit(Debye_graphing_3, temperatures, 
                      heat_capacities, 
                      bounds=(0,[np.inf, np.inf, np.inf, 7, 7, 7, 7])
                     )
Td_fit = pars[0]; Td2_fit = pars[1]; Td3_fit = pars[2] 
atoms_fit = pars[3]; a1_fit = pars[4]; a2_fit = pars[5]; a3_fit = pars[6]

plt.scatter(temperatures, heat_capacities, s=10, c='r', label='Mystery Data')
plt.plot(temperatures, 
         Debye_graphing_3(
             temperatures, 
             Td_fit, Td2_fit, Td3_fit, 
             atoms_fit, a1_fit, a2_fit, a3_fit),
         alpha=1, lw=2, c='k', label='Best Fit')

plt.title('Fitting with a Sum of Three Debye Models')
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity (J/K)')
plt.legend() 

# Computing square root of sum of squared residuals and printing result
D3_res = LSRL(heat_capacities, Debye_graphing_3(temperatures, 
                                                Td_fit, Td2_fit, Td3_fit, 
                                                atoms_fit, 
                                                a1_fit, a2_fit, a3_fit))
print('Sqrt of Sum of Squared Residuals : {:0.3f}'.format(D3_res))
print('Sum of Coefficients in Series = {:0.3f}'.format(atoms_fit*(a1_fit+a2_fit+a3_fit)))

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

plt.figure()

pars, cov = curve_fit(
    Debye_graphing_3_alt, temperatures, 
    mod_heat_capacities, 
    bounds=(0,[np.inf, np.inf, np.inf, 7, 7, 7, 7])
)
Td_fit = pars[0]; Td2_fit = pars[1]; Td3_fit = pars[2] 
atoms_fit = pars[3]; a1_fit = pars[4]; a2_fit = pars[5]; a3_fit = pars[6]

plt.scatter(temperatures, mod_heat_capacities, s=10, c='r', label='Mystery Data')
plt.plot(temperatures, 
         Debye_graphing_3_alt(
             temperatures, 
             Td_fit, Td2_fit, Td3_fit, 
             atoms_fit, a1_fit, a2_fit, a3_fit),
         alpha=1, lw=2, c='k', label='Best Fit')

plt.title('Fitting with a Sum of Three Debye Models, Over Temp Cubed')
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity (J/K)')
plt.legend() 

# Computing square root of sum of squared residuals and printing result
D3A_res = LSRL(mod_heat_capacities, Debye_graphing_3_alt(temperatures, 
                                                Td_fit, Td2_fit, Td3_fit,
                                                atoms_fit,
                                                a1_fit, a2_fit, a3_fit))
print('Sqrt of Sum of Squared Residuals : {:f}'.format(D3A_res*1000))

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

%matplotlib inline

plt.figure()

# Fitting the data and unpacking the fit parameters
pars2E, cov2E = curve_fit(
    Einstein_graphing_2, temperatures, 
    heat_capacities, 
    bounds=(0,[np.inf, np.inf, 7, 7, 7])
)
Te_fit2E, Te2_fit2E, atoms_fit2E, a1_fit2E, a2_fit2E = pars2E
        
# Plotting
plt.scatter(temperatures, heat_capacities, s=10, c='r', label='Mystery Data')
plt.plot(temperatures, 
         Einstein_graphing_2(temperatures, Te_fit2E, Te2_fit2E, atoms_fit2E, a1_fit2E, a2_fit2E), 
         alpha=1, lw=2, c='k', label='Best Fit')

plt.title('Fitting with a Sum of Two Einstein Models')
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity (J/K)')
plt.legend() 

# Computing square root of sum of squared residuals and printing result
E2_res = LSRL(heat_capacities, Einstein_graphing_2(temperatures, 
                                                Te_fit2E, Te2_fit2E,
                                                atoms_fit2E,
                                                a1_fit2E, a2_fit2E))
print('Sqrt of Sum of Squared Residuals = {:0.3f}'.format(E2_res))
print('Sum of Coefficients in Series = {:0.3f}'.format(atoms_fit2E*(a1_fit2E+a2_fit2E)))

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

plt.figure()

# Fitting the data and unpacking the fit parameters
pars2EA, cov2EA = curve_fit(
    Einstein_graphing_2_alt, temperatures, 
    mod_heat_capacities, 
    bounds=(0,[np.inf, np.inf, 7, 7, 7])
)
Te_fit2EA, Te2_fit2EA, atoms_fit2EA, a1_fit2EA, a2_fit2EA = pars2EA 
        
# Plotting
plt.scatter(temperatures, mod_heat_capacities, s=10, c='r', label='Mystery Data')
plt.plot(temperatures, 
         Einstein_graphing_2_alt(temperatures, 
                                 Te_fit2EA, Te2_fit2EA, 
                                 atoms_fit2EA, 
                                 a1_fit2EA, a2_fit2EA), 
         alpha=1, lw=2, c='k', label='Best Fit')

plt.title('Fitting with a Sum of Two Einstein Models Over Temperature Cubed')
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity over Temp Cubed')
plt.legend()

# Computing square root of sum of squared residuals and printing result
E2A_res = LSRL(mod_heat_capacities, Einstein_graphing_2_alt(temperatures, 
                                                Te_fit2EA, Te2_fit2EA,
                                                atoms_fit2EA,
                                                a1_fit2EA, a2_fit2EA))
print('Sqrt of Sum of Squared Residuals : {:f}'.format(E2A_res*1000))

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

%matplotlib inline

plt.figure()

# Fitting the data and unpacking the fit parameters
pars2ED, cov2ED = curve_fit(
    Model_graph_2, temperatures, 
    heat_capacities, 
    bounds=(0,[np.inf, np.inf, 7, 7, 7])
)
Td_fit2ED, Td2_fit2ED, atoms_fit2ED, a1_fit2ED, a2_fit2ED = pars2ED
        
# Plotting
plt.scatter(temperatures, heat_capacities, s=10, c='r', label='Mystery Data')
plt.plot(temperatures, 
         Model_graph_2(temperatures, Td_fit2ED, Td2_fit2ED, 
                       atoms_fit2ED, 
                       a1_fit2ED, a2_fit2ED), 
         alpha=1, lw=2, c='k', label='Best Fit')

plt.title('Fitting with a Sum of Debye and Einstein Models')
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity (J/K)')
plt.legend() 

# Computing square root of sum of squared residuals and printing result
ED2_res = LSRL(heat_capacities, Model_graph_2(temperatures, 
                                                Td_fit2ED, Td2_fit2ED,
                                                atoms_fit2ED,
                                                a1_fit2ED, a2_fit2ED))
print('Sqrt of Sum of Squared Residuals : {:0.3f}'.format(ED2_res))
print('Sum of Coefficients in Series = {:0.3f}'.format(atoms_fit2ED*(a1_fit2ED+a2_fit2ED)))

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

plt.figure()

parsM2A, covM2A = curve_fit(
    Mult_graph_2_alt, temperatures, 
    mod_heat_capacities,
    bounds=(0,[np.inf, np.inf, 7, 7, 7])
)
Td_fitM2A, Td2_fitM2A, atoms_fitM2A, a1_fitM2A, a2_fitM2A = parsM2A

plt.scatter(temperatures, mod_heat_capacities, s=10, c='r', label='Mystery Data')
plt.plot(temperatures, 
         Mult_graph_2_alt(
             temperatures, 
             Td_fitM2A, Td2_fitM2A, 
             atoms_fitM2A, a1_fitM2A, a2_fitM2A),
         alpha=1, lw=2, c='k', label='Best Fit')

plt.title('Fitting with a Sum of Debye and Einstein Models, Over Temp Cubed')
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity Over Temp Cubed')
plt.legend() 

# Computing square root of sum of squared residuals and printing result
M2A_res = LSRL(mod_heat_capacities, Mult_graph_2_alt(temperatures, 
                                                Td_fitM2A, Td2_fitM2A,
                                                atoms_fitM2A,
                                                a1_fitM2A, a2_fitM2A))
print('Sqrt of Sum of Squared Residuals : {:f}'.format(M2A_res*1000))

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

plt.figure()

pars2D1E, cov2D1E = curve_fit(
    Model_3E1_graphing, temperatures, 
    heat_capacities, 
    bounds=(0,[np.inf, np.inf, np.inf, 7, 7, 7, 7])
)
Td_fit2D1E, Td2_fit2D1E, Td3_fit2D1E, atoms_fit2D1E, a1_fit2D1E, a2_fit2D1E, a3_fit2D1E = pars2D1E

plt.scatter(temperatures, heat_capacities, s=10, c='r', label='Mystery Data')
plt.plot(temperatures, 
         Model_3E1_graphing(
             temperatures, 
             Td_fit2D1E, Td2_fit2D1E, Td3_fit2D1E, 
             atoms_fit2D1E, a1_fit2D1E, a2_fit2D1E, a3_fit2D1E),
         alpha=1, lw=2, c='k', label='Best Fit')

plt.title('Fitting with a Sum of Two Debye Models and One Einsten Model')
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity (J/K)')
plt.legend() 

# Computing square root of sum of squared residuals and printing result
D3A_res = LSRL(heat_capacities, Model_3E1_graphing(temperatures, 
                                                Td_fit2D1E, Td2_fit2D1E, Td3_fit2D1E,
                                                atoms_fit2D1E,
                                                a1_fit2D1E, a2_fit2D1E, a3_fit2D1E))
print('Sqrt of Sum of Squared Residuals : {:f}'.format(D3A_res))

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

plt.figure()

pars2D1EA, cov2D1EA = curve_fit(
    Model_graphing_3E1_alt, temperatures, 
    mod_heat_capacities, 
    bounds=(0,[np.inf, np.inf, np.inf, 7, 7, 7, 7])
)
Td_fit2D1EA, Td2_fit2D1EA, Td3_fit2D1EA, atoms_fit2D1EA, a1_fit2D1EA, a2_fit2D1EA, a3_fit2D1EA = pars2D1EA

plt.scatter(temperatures, mod_heat_capacities, s=10, c='r', label='Mystery Data')
plt.plot(temperatures, 
         Model_graphing_3E1_alt(
             temperatures, 
             Td_fit2D1EA, Td2_fit2D1EA, Td3_fit2D1EA, 
             atoms_fit2D1EA, a1_fit2D1EA, a2_fit2D1EA, a3_fit2D1EA),
         alpha=1, lw=2, c='k', label='Best Fit')

plt.title('Fitting with a Sum of Two Debye Models and One Einsten Model, Over Temp Cubed')
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity (J/K)')
plt.legend() 

# Computing square root of sum of squared residuals and printing result
D2E1A_res = LSRL(mod_heat_capacities, Model_graphing_3E1_alt(temperatures, 
                                                Td_fit2D1EA, Td2_fit2D1EA, Td3_fit2D1EA,
                                                atoms_fit2D1EA,
                                                a1_fit2D1EA, a2_fit2D1EA, a3_fit2D1EA))
print('Sqrt of Sum of Squared Residuals : {:f}'.format(D2E1A_res*1000))

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

plt.figure()

pars2E1D, cov2E1D = curve_fit(
    Model_3D1_graphing, temperatures, 
    heat_capacities, 
    bounds=(0,[np.inf, np.inf, np.inf, 7, 7, 7, 7])
)
Td_fit2E1D, Td2_fit2E1D, Td3_fit2E1D, atoms_fit2E1D, a1_fit2E1D, a2_fit2E1D, a3_fit2E1D = pars2E1D

plt.scatter(temperatures, heat_capacities, s=10, c='r', label='Mystery Data')
plt.plot(temperatures, 
         Model_3D1_graphing(
             temperatures, 
             Td_fit2E1D, Td2_fit2E1D, Td3_fit2E1D, 
             atoms_fit2E1D, a1_fit2E1D, a2_fit2E1D, a3_fit2E1D),
         alpha=1, lw=2, c='k', label='Best Fit')

plt.title('Fitting with a Sum of One Debye Model and Two Einsten Models')
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity (J/K)')
plt.legend() 

# Computing square root of sum of squared residuals and printing result
E2A_res = LSRL(heat_capacities, Model_3D1_graphing(temperatures, 
                                                Td_fit2E1D, Td2_fit2E1D, Td3_fit2E1D,
                                                atoms_fit2E1D,
                                                a1_fit2E1D, a2_fit2E1D, a3_fit2E1D))
print('Sqrt of Sum of Squared Residuals : {:f}'.format(D3A_res))

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

plt.figure()

pars2E1DA, cov2E1DA = curve_fit(
    Model_graphing_3D1_alt, temperatures, 
    mod_heat_capacities, 
    bounds=(0,[np.inf, np.inf, np.inf, 7, 7, 7, 7])
)
Td_fit2E1DA, Td2_fit2E1DA, Td3_fit2E1DA, atoms_fit2E1DA, a1_fit2E1DA, a2_fit2E1DA, a3_fit2E1DA = pars2E1DA

plt.scatter(temperatures, mod_heat_capacities, s=10, c='r', label='Mystery Data')
plt.plot(temperatures, 
         Model_graphing_3D1_alt(
             temperatures, 
             Td_fit2E1DA, Td2_fit2E1DA, Td3_fit2E1DA, 
             atoms_fit2E1DA, a1_fit2E1DA, a2_fit2E1DA, a3_fit2E1DA),
         alpha=1, lw=2, c='k', label='Best Fit')

plt.title('Fitting with a Sum of One Debye Model and Two Einsten Models, Over Temp Cubed')
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity (J/K)')
plt.legend() 

# Computing square root of sum of squared residuals and printing result
E2D1A_res = LSRL(mod_heat_capacities, Model_graphing_3D1_alt(temperatures, 
                                                Td_fit2E1DA, Td2_fit2E1DA, Td3_fit2E1DA,
                                                atoms_fit2E1DA,
                                                a1_fit2E1DA, a2_fit2E1DA, a3_fit2E1DA))
print('Sqrt of Sum of Squared Residuals : {:f}'.format(E2D1A_res*1000))

