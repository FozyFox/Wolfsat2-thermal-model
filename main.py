## This is a Python representation of the physics behind a 0D thermal model for a cubesat in space.
"""
LEO Thermal Model
Created by Ekansh Kumar Sharma for Wolfsat-2.
Date: April 16 - May 2, 2026

This is a Python representation of the physics behind a 0D thermal model for a CubeSat in space.

Notes:
This model includes:
- 6 External faces + 1 Internal thermal node
- Spin-averaged solar flux using a time-dependent 3D rotation matrix
- Direct solar heating (+X), Earth IR (-Z), and albedo (-Z) contributions
- Radiative cooling via Stefan-Boltzmann law
- Conduction between adjacent faces and the internal node
- Internal heat generation (estimated)
- Euler integration over multiple orbits to reach steady state temperatures
- Steady-state max/min of each face and internal node
- Basic anisotropic thermal expansion for birch wood chassis


- All physics equations, logic, and modeling were developed (or researched) by the author.
- Artificial Intelligience was used in order to debug and fix errors in the code.
- The simulation is intended for early-phase thermal analysis of Wolfsat-2.

Mission:
The Wolfsat-2 is is a 1U CubeSat mission designed to test the viability of using a wood chassis rather than the usual metal.
It operates in a very low Earth orbit (~230 km). The orbit is effectively circular for thermal purposes.
This model estimates both the RPM as constant and a pre-determined eclipse percentage.
The author has also developed a drag-decay model for Wolfsat-2 to estimate a lifespan for the satellite at https://github.com/FozyFox/LEO-drag-decay-model

We consider the sun_fraction as an estimate because finding the exact value for it is overkill for a CubeSat mission.
Besides, it's dependent on the orbit's beta angle, which varies with both RAAN and the time of the year.

The theoretical range at this altitude, inclination, and time of launch:
    - Cold-case (max eclipse-ish): ~0.60 sunlit
    - Hot-case (no eclipse): 1.00 sunlit
We'll use a representative mid-case value:
    sun_fraction = 0.75


This is a multi-node thermal model with anisotropic thermal expansion.
"""


## Imports
import numpy as np
import matplotlib.pyplot as plt

## Constants (Orbit and Cubesat)
# Orbital Period (s)
P_orbit = 90 * 60

# Solar Absoptivity
alpha = 0.40

# IR Emissivity
epsilon = 0.53

# Density of Birch (kg m^-3)
rho = 650

# Specific Heat of Birch (J kg^-1 K^-1)
c_birch = 1300

## Area (m^2)
# Lengths (m)
Lx_panel = 0.1
Ly_panel = 0.1

A = Lx_panel * Ly_panel

# Panel Thickness (m)
Lz_panel = 0.006

## Some other Constants
# Stefan-Boltzmann Constant (W m^-2 K^-)4
sigma = 5.670374419e-8

# Solar constant (W m^-2)
S_sun = 1361

# Approximating albedo flux (W m^-2)
albedo = 150

# Approximating Earth IR flux (W m^-2)
IR_earth = 237

## Estimating some other stuff
# Mass (kg)
m_panel = rho * A * Lz_panel

# (J K^-1)
C_external = m_panel * c_birch

# Sunlight fraction
sun_fraction = 0.75 # 75% sunlight

## Spin
# RPM
RPM_spin = 3 # RPM
# Spin Period (s)
P_spin = 60 / RPM_spin
omega_spin = 2 * np.pi / P_spin


## Functions
# Sunlight/Eclipse function
def sunlit(t):
    phase = (t % P_orbit) / P_orbit
    return 1.0 if phase < sun_fraction else 0.0

# Now the actual math
def derivatives(t, T_faces, T_internal):

    # Sunlight/eclipse state
    sun_state = sunlit(t)

    # Array for the 6 faces
    dT_faces = np.zeros(6)

    # Face normals (orientation of each face)
    face_normals = np.array([
        [ 1,  0,  0], # +X face (0)
        [-1,  0,  0], # -X face (1)
        [ 0,  1,  0], # +Y face (2)
        [ 0, -1,  0], # -Y face (3)
        [ 0,  0,  1], # +Z face (4)
        [ 0,  0, -1], # -Z face (5)
    ])

    ## Spin kinematics: spin about body Y-axis
    theta = omega_spin * t  # angle [rad]

    ## 3D rotation matrix for spin about the y-axis
    Ry = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0            , 1,             0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    ## Sun vectors
    # Sun direction vector (this is what controls the "spinning": important for spinning)
    sun_direction = np.array([1, 0 , 0]) # +X face

    sun_vector = Ry @ sun_direction
    sun_vector = sun_vector / np.linalg.norm(sun_vector) # Dividing sun_vector by the magnitude

    # Cosine value for each of the panels
    Sun_cosine = face_normals @ sun_vector

    # So that we don't have "negative" values
    Sun_cosine = np.maximum(Sun_cosine, 0.0)

    ## Earth vectors
    # Earth direction vector (this is also what controls "spinning": important for spinning)
    earth_direction = np.array([0, 0 , -1]) # -Z Face

    earth_vector = Ry @ earth_direction
    earth_vector = earth_vector / np.linalg.norm(earth_vector) # Dividing earth_vector by the magnitude

    # Cosine values for each of the panels
    Earth_cosine = face_normals @ earth_vector

    # So that we don't have "negative" values
    Earth_cosine = np.maximum(Earth_cosine, 0.0)

    ## Conduction stuff
    # Nearby faces list for conduction (What faces touch each other)
    Nearby_faces = {
        0: [2, 3, 4, 5],
        1: [2, 3, 4, 5],
        2: [0, 1, 4, 5],
        3: [0, 1, 4, 5],
        4: [0, 1, 2, 3],
        5: [0, 1, 2, 3],
    }

    # Thermal conductance between faces (W K^-1)
    K_conduction = 0.25
    # Thermal conductance - internal components (estimated) (W K^-1)
    K_internal = 0.15
    # Specific Heat of internal components (estimated) (J K^-1)
    C_internal = 500

    # Loop for each of the 6 panels
    for i in range(6):
        # Heat inputs (Q) for face "i"
        Q_sun = alpha * S_sun * A * (sun_state * Sun_cosine[i])
        Q_albedo = alpha * albedo * A * Earth_cosine[i] * sun_state      # We'll consider sun_state on for now
        Q_IR = epsilon * IR_earth * A  * Earth_cosine[i]                 # Always present, so no "sun_state"
        Q_radiation = epsilon * sigma * A * (T_faces[i] ** 4)

        # Conduction between the panels
        Q_conduction = 0.0
        for j in Nearby_faces[i]:
            Q_conduction += K_conduction * (T_faces[j] - T_faces[i])

        # Internal conduction
        Q_internal_conduction = K_internal * (T_internal - T_faces[i])

        # Net Heat
        Q_net = Q_sun + Q_albedo + Q_IR + Q_conduction + Q_internal_conduction - Q_radiation

        # dT/dt for external (chassis)
        dT_faces[i] = Q_net / C_external



    # Battery internal heat (W)
    Q_internal_node = 3.0 # Watts, total internal heat (electronics)
    if sun_state == 1.0:
        Q_internal_node += 0.5 # charging heat (battery loss) (just estimating)
    else:
        Q_internal_node += 0.2 # discharging heat (also battery loss) (just estimating)
    
    # Internal Node Derivative
    Q_internal_total = 0.0
    for i in range(6):
        Q_internal_total += K_internal * (T_faces[i] - T_internal)


    # Adding internal battery heat
    Q_internal_total += Q_internal_node
        
    dTi_dt = Q_internal_total / C_internal
    
    return dT_faces, dTi_dt
    

def propagated_temp():

    ## Setup variables
    # Timestep (s)
    dt = 1
    # Starting time
    t = 0
    
    # Orbits calculated
    n_orbits = 10

    # Initial Temperature of External Chassis faces (K) 
    T_faces = np.ones(6) * 280 # Considering 280 K for now. Probably gonna change it

    # Initial Temperature of internal environment (K)
    T_internal = 280
    
    # Lists
    Te_list = []
    Ti_list = []
    t_list = []

    # Euler Integration
    while (t < n_orbits * P_orbit):
        # Derivatives of 
        dT_faces, dTi_dt = derivatives(t, T_faces, T_internal)

        T_faces += dT_faces * dt
        T_internal += dTi_dt * dt
        t += dt

        Te_list.append(T_faces.copy())
        Ti_list.append(T_internal)
        t_list.append(t)

    return np.array(t_list), np.array(Te_list), np.array(Ti_list), n_orbits


## Thermal Expansion function (using steady state temperatures because we don't need a full simulation for a cubesat)
# Firstly: birch is anisotropic
# It has different values across and along the grain

def thermal_expansion(Lx_panel, Ly_panel, Lz_panel, Te_max, Te_min):
    ## Thermal Expansion variables (K^-1)
    alpha_across = 30e-6
    alpha_along = 3e-6

    # Lx and Ly of the panels are area (10 cm each)
    # Lz of the panels is the thickness (6 mm)

    # Variable for Temperature during building (when the material is the "least" stressed as a reference)
    T_build = 297.15 # K (or 24 °C)

    # Array for each of the 6 panels values
    dLx_list = np.zeros(6)
    dLy_list = np.zeros(6)
    dLz_list = np.zeros(6)


    print(f"\nThermal Expansion (relative to {T_build - 273.15} °C build temperature)")

    # Loop for each of the panels (because they each receive a different temperature)
    for i in range(6):
        dT = Te_max[i] - T_build

        # Calculating each of the panels
        dLx = alpha_across * Lx_panel * dT
        dLy = alpha_across * Ly_panel * dT
        dLz = alpha_along * Lz_panel * dT

        # Adding the values to the list
        dLx_list[i] = dLx
        dLy_list[i] = dLy
        dLz_list[i] = dLz

        print(f"Panel {i+1}")
        print(f"    ΔT = {dT:.3f} K")
        print(f"    ΔLx = {dLx*1e3:.6f} mm")
        print(f"    ΔLy = {dLy*1e3:.6f} mm")
        print(f"    ΔLz = {dLz*1e3:.6f} mm")

    print("\nDifferential Expansion Analysis:")
    print(f"\nDifference in X-panels (panels 1 and 2 (or 0 and 1 in the model) = face2 - face1):")
    print(f"    x-expansion: {(dLx_list[1] - dLx_list[0])*1e3:.6f} mm")
    print(f"    y-expansion: {(dLy_list[1] - dLy_list[0])*1e3:.6f} mm")
    print(f"    z-expansion: {(dLz_list[1] - dLz_list[0])*1e3:.6f} mm")
    print(f"\nDifference in Y-panels (panels 3 and 4 (or 2 and 3 in the model) = face4 - face3):")
    print(f"    x-expansion: {(dLx_list[3] - dLx_list[2])*1e3:.6f} mm")
    print(f"    y-expansion: {(dLy_list[3] - dLy_list[2])*1e3:.6f} mm")
    print(f"    z-expansion: {(dLz_list[3] - dLz_list[2])*1e3:.6f} mm")
    print(f"\nDifference in Z-panels (panels 5 and 6 (or 4 and 5 in the model) = face6 - face5):")
    print(f"    x-expansion: {(dLx_list[5] - dLx_list[4])*1e3:.6f} mm")
    print(f"    y-expansion: {(dLy_list[5] - dLy_list[4])*1e3:.6f} mm")
    print(f"    z-expansion: {(dLz_list[5] - dLz_list[4])*1e3:.6f} mm")

    return dLx_list, dLy_list, dLz_list


# Actually running TEMPERATURE LOOP (NOT THERMAL EXPANSION)
t_list, Te_list, Ti_list, n_orbits = propagated_temp()


## Finding Steady-state temperature using the final orbit
# Time where final orbit begins
ss_start = (n_orbits - 1) * P_orbit

# Final orbit points
ss_points = np.where(t_list >= ss_start)

Te_ss = Te_list[ss_points] # External faces
Ti_ss = Ti_list[ss_points] # Internal

# Finding max and mind of each
Te_max = Te_ss.max(axis = 0) # "axis = 0" returns max temps for each face
Te_min = Te_ss.min(axis = 0)

'''
Te_list is like follows:

time_1 = [temp_face_0, temp_face_1, temp_face_2, temp_face_3, temp_face_4, temp_face_5]
time_2 = [temp_face_0, temp_face_1, temp_face_2, temp_face_3, temp_face_4, temp_face_5]
time_3 = [temp_face_0, temp_face_1, temp_face_2, temp_face_3, temp_face_4, temp_face_5]
... onwards

axis = 0 finds the max of each column, and returns: [max_temp_0, max_temp_1, max_temp_2, max_temp_3, max_temp_4, max_temp_5]
I had to search this up to figure out how to do it too.
'''

Ti_max = Ti_ss.max() ## This is only one list, so it doesn't need "axis = 0"
Ti_min = Ti_ss.min()

# Printing the steady-state temperatures
print("\nSteady-State External Panels:")
for i in range(6):
    print(f"Panel {i+1}: min = {Te_min[i]:.3f} K "
          f"(or {(Te_min[i] - 273.15):.3f} °C); "
          f"max = {Te_max[i]:.3f} K "
          f"(or {(Te_max[i] - 273.15):.3f} °C)."
          )

print("\nSteady State Internal:")
print(f"Internal: min = {Ti_min:.3f} "
      f"(or {(Ti_min - 273.15):.3f} °C) "
      f"K; max = {Ti_max:.3f} K "
      f"(or {(Ti_max - 273.15):.3f} °C).\n"
      )
        
# Running the Thermal Expansion loop
dLx_list, dLy_list, dLz_list = thermal_expansion(Lx_panel, Ly_panel, Lz_panel, Te_max, Te_min)



### Plots
## Main graph
plt.figure
plt.title(f"Temperature over time for {n_orbits} orbits")

# Setting time list to numpy list for ease
time_constant = 3600 # seconds -> hours

# Shade sunlight and eclipse zones
for n in range(n_orbits):  # number of orbits
    t_start = (n * P_orbit) / time_constant
    t_sun_end = t_start + (sun_fraction * P_orbit) / time_constant # T_start is already in hours
    t_eclipse_end = t_start + (P_orbit) / time_constant

    if n == 0:
        # First orbit gets labeled on the legend
        # Sunlit region (yellow)
        plt.axvspan(t_start, t_sun_end, color='yellow', alpha=0.15, label="Sunlit")

        # Eclipse region (gray)
        plt.axvspan(t_sun_end, t_eclipse_end, color='gray', alpha=0.15, label="Eclipse")
    else:
        # All other orbits do not get labled
        # Sunlit region (yellow)
        plt.axvspan(t_start, t_sun_end, color='yellow', alpha=0.15)

        # Eclipse region (gray)
        plt.axvspan(t_sun_end, t_eclipse_end, color='gray', alpha=0.15)


# Plot for faces
for i in range(6):
    plt.plot(t_list / time_constant, [Te[i] for Te in Te_list], label=f"External Panel {i+1}")
    
# Plot for internal temp
plt.plot(t_list / time_constant, Ti_list, label="Internal")
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

plt.xlabel("Time (hours)")
plt.ylabel("Temperature (K)")

# 0 C line
plt.axhline(273.15, color='blue', linestyle='--', linewidth=1, label='0°C (273.15 K)')

# Show the legend BEFORE showing the second axis (causes an error otherwise)
plt.legend()

# Second axis in Celsius
ax = plt.gca()
ax2 = ax.twinx()

ymin, ymax = ax.get_ylim()
ax2.set_ylim(ymin - 273.15, ymax - 273.15)

ax2.set_ylabel("Temperature (°C)")


## Plotting the last orbit
t_last = t_list[ss_points]
Te_last = Te_list[ss_points]
Ti_last = Ti_list[ss_points]

plt.figure()
plt.title("Last Orbit Temperature over time")

# Sunlight/eclipse
t_last_start = ss_start / time_constant
t_sun_end_last = t_last_start + (sun_fraction * P_orbit) / time_constant
t_eclipse_end_last = t_last_start + (P_orbit) / time_constant

# Sunlit/eclipse plotting
plt.axvspan(t_last_start, t_sun_end_last, color="yellow", alpha=0.15, label="Sunlit")
plt.axvspan(t_sun_end_last, t_eclipse_end_last, color="gray", alpha=0.15, label="Eclipse")

# Plot external faces
for i in range(6):
    plt.plot(t_last / time_constant, Te_last[:, i], label=f"External Panel {i+1}")

# Plot internal
plt.plot(t_last / time_constant, Ti_last, label="Internal")

plt.xlabel("Time (hours)")
plt.ylabel("Temperature (K)")
plt.grid(True, linestyle="--", alpha=0.6)


# 0 C line
plt.axhline(273.15, color='blue', linestyle='--', linewidth=1, label='0°C (273.15 K)')

# Show the legend before 0 C line
plt.legend()

# Second axis in Celsius
ax = plt.gca()
ax2 = ax.twinx()

ymin, ymax = ax.get_ylim()
ax2.set_ylim(ymin - 273.15, ymax - 273.15)

ax2.set_ylabel("Temperature (°C)")


# Show the graph
plt.show()

# :)
