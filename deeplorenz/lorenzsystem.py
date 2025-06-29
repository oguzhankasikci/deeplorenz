import numpy as np
from scipy.integrate import odeint
import pandas as pd
import os


class LorenzClass:
    def __init__(self, 
                params = (10.0, 8.0 / 3.0, 28.0),
                X0 = (0.0, 1.0, 20.0),
                t_span = (0.0, 50.0),
                n_steps  = 50_000
                ):
        # Import Plotter class for plotting methods 
        from lplotter import Plotter
        
        """ Construct parameters and class variables"""
        # Lorenz system parameters
        self.sigma = float(params[0])
        self.beta = float(params[1])
        self.rho = float(params[2])

        # Initial state, (x,y,z) = (X0[0],X0[1],X0[2])
        self.X0 = np.asarray(X0, dtype=float)        
        
        # Time grid 
        self.t = np.linspace(float(t_span[0]), float(t_span[1]), int(n_steps), endpoint=True)
        self.dt = np.insert(np.diff(self.t), 0, self.t[1] - self.t[0])
        

        # Solution invalidate
        self._solution = None  # shape (n_steps,3)  

        ## Plotter instance for plotting methods
        self.plotter = Plotter(self)


    ######################### Class Methods #################

    
    #=== Update class parameters, initial value and time interval ==============
    def set_parameters(self, params = None):
        """  Update Parameters if only those provided"""
        if params is not None:
            self.sigma = float(params[0])
            self.beta = float(params[1])
            self.rho = float(params[2])
        self._solution = self.solve()  ## reset the solution
        self.plotter.__init__(self)  # Reinitialize the plotter with new parameters

    def set_time_grid(self, t0, tf, n_steps):
        """ Redefine the time array t and dt."""
        self.t = np.linspace(float(t0), float(tf), int(n_steps), endpoint=True)
        self.dt = np.insert(np.diff(self.t), 0, self.t[1] - self.t[0])
        self._solution =  self.solve()  ## reset the solution
        self.plotter.__init__(self)  # Reinitialize the plotter with new parameters

    def set_initial_value(self, X0):
        self.X0 = np.asarray(X0, dtype=float) 
        self._solution =  self.solve() ## reset the solution
        self.plotter.__init__(self)  # Reinitialize the plotter with new parameters

    ## =========================================================

    # Dynamic Equations 
    def lorenz(self, vars, time=0):
        """ Lorenz vector field (dx/dt, dy/dt, dz/dt)."""
        x, y, z = vars
        a, b, c = self.sigma, self.beta, self.rho
        dx = a * (y - x)
        dy = x * (c - z) - y
        dz = x * y - b * z
        return np.array([dx, dy, dz])

    def solve(self, rtol=1e-12, atol_scale=1e-12):
        """ Integrate the ODE system and store the result as shape(n_steps,3)"""
        atol = atol_scale * np.ones_like(self.X0)
        self._solution = odeint(
            self.lorenz, self.X0, self.t, rtol=rtol, atol=atol
        )
        return self._solution
    
    def components(self):
        """ Provide the solution components as dictionary ('x': x_val, 'y': y_val, 'z': z_val))"""
        sol = self._solution
        return dict(x= sol[:,0],y= sol[:,1],z= sol[:,2])
    

    ## ===  Plotting Methods ===
    def __getattr__(self, name: str):
        """ Redirects to the Plotter instance for plotting methods."""
        if hasattr(self.plotter, name):
            return getattr(self.plotter, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    ## ===  Writing to a  'csv' file ===
    def write_csv(self, filename='lorenz_sol.csv'):
        """ Write the solution to a CSV file."""         
        full_data = np.c_[self.t, self._solution]       
        df = pd.DataFrame(full_data, columns=['Time','X', 'Y', 'Z'])

        # If there is file with same name change the name
        for fname in os.listdir('.'):
            if fname.endswith('.csv'):  
                filename = fname.replace('.csv', '_new.csv')
                df.to_csv(filename, index=False)
                print(f"File {filename} already exists. Renaming to {filename}.")
                break            
         
        # Write to csv
        df.to_csv(filename, index=False)
        return print(f"Solution written to {filename} successfully.")
    
    # ===  Reading from a 'csv' file ===
    # Note: This method assumes the CSV file has columns 'Time', 'X', 'Y', 'Z'
    # and that the data is in the same format as produced by write
    def read_csv(self, filename='lorenz_sol.csv'):
        """ Read the solution from a CSV file."""
        df = pd.read_csv(filename)
        self.t = df['Time'].values
        self._solution = df[['X', 'Y', 'Z']].values
        self.plotter._solution = self._solution # Reinitialize the plotter with new data
        self.plotter.t = self.t  # Reinitialize the plotter with new data
        return print(f"Solution read from {filename} successfully.")


# Example usage ---------------------------------------------------------
if __name__ == "__main__":
    
    lc = LorenzClass()
    # lc.solve()          # integrates with defaults
    # lc.plot_2d()        # 2‑D component plot
    # lc.plot_3d()        # 3‑D trajectory  
    lc.write_csv()      # write solution to csv file
    # lc.plot_phase2d('xz')  # 2d phase plot for x and y)    
    # lc.plotcomp2d('x')  # plot x component over time   
    

    # # # Update  initial value and parameters, then solve again and plot 
    # lc.set_initial_value(X0=(20.0, 1.0, 1.0))
    # lc.set_parameters(params= (5.0, 6.0 / 3.0, 1.0))
    # lc.set_time_grid(0, 30, n_steps=30_000)
    # lc.plot_2d()    
    # lc.plot_3d()
    # lc.plotcomp2d('z')  # plot z component over time
    # lc.plot_phase2d('xz')
    # lc.write_csv()

    # # # Read from csv file and plot
    # lc.read_csv('lorenz_sol.csv')
    # lc.plot_phase2d('xz')
    # lc.plot_3d()

    
