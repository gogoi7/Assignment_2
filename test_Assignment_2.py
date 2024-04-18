import numpy as np
from Assignment_2 import ODESolver

def test_odesolver_init():
    # Test valid initialization
    f = lambda y, t: -y
    y_0 = 1.0
    t_0 = 0.0
    t_f = 10.0
    dt = 0.01

    solver = ODESolver(f, y_0, t_0, t_f, dt)

    assert solver.f == f
    assert solver.y_0 == y_0
    assert solver.t_0 == t_0
    assert solver.t_f == t_f
    assert solver.dt == dt

    # Test invalid initialization (t_f <= t_0)
    t_f = -1.0

    try:
        solver = ODESolver(f, y_0, t_0, t_f, dt)
        assert False  # This line should not be reached
    except ValueError as e:
        assert str(e) == "Stop time must be greater than start time."

def test_odesolver_solve():
    # Define the ODE
    f = lambda y, t: -y

    # Set the parameters
    y_0 = 1.0
    t_0 = 0.0
    t_f = 10.0
    dt = 0.01

    # Initialize the ODESolver
    solver = ODESolver(f, y_0, t_0, t_f, dt)

    # Solve the ODE
    numerical_solution = solver.solve()

    # Generate the time array
    t = np.arange(t_0, t_f, dt)

    # Calculate the analytical solution
    analytical_solution = y_0 * np.exp(-t)

    # Compare the numerical solution to the analytical solution
    assert np.allclose(numerical_solution, analytical_solution)