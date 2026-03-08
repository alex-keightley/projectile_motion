import numpy as np
import matplotlib.pyplot as plt


# Class Object_Model
# This class constructs an object with several class variables
#     mass                - the mass of the object in kg
#     radius              - the non-zero radius of an object - object exists in R2 and is not point-like
#     cs_area             - cross-sectional area of the object to be used in drag calculations
#     drag_coeff          - the drag coefficient of the object where 0 <= drag_coeff <= 1
#     position_array      - an array to hold the 2D position of the object, to be appended upon each step of 
#                            Euler's method
#     velocity_array      - an array to hold the 2D velocity vector of the object, to be appended upon each step
#                            of Euler's method
#     force_due_to_impulse_array - an array to hold the force due to impulse upon each collision. The force is calculated
#                            via the Impulse-momentum formula, from the change in momentum of the object and the delta_t
#                            is taken to be the time-step of Euler's method (note: the time-step then is required to be 
#                            chosen such that it is a reasonable approximation for the duration of impact of the objects
#     model_gravity       - acceleration constant due to the Force of gravity for the model in which the object operates.
#     total_energy        - An array of 2D vectors in which the first index is the kinetic energy of the object at each iteration
#                            and the second index is the gravitational potential energy of the object at each iteration.

class Object_Model:
    def __init__(self, init_mass=1.0, init_x_position=0.0, init_y_position=0.5, init_x_velocity=0.0, init_y_velocity=0.0, init_radius=1.0, init_gravity = -9.81, init_drag_coeff = 0):
        self.mass = init_mass
        self.radius = init_radius
        self.cs_area = np.pi * self.radius**2
        self.drag_coeff = init_drag_coeff
        self.position_array = [[init_x_position, init_y_position]]
        self.velocity_array = [[init_x_velocity, init_y_velocity]]
        self.force_due_to_impulse_array = []
        self.model_gravity = init_gravity
        self.total_energy = [[(1/2) * self.mass * np.linalg.norm(self.velocity_array[-1])**2, self.mass * np.abs(self.model_gravity) * self.position_array[-1][1]]]

    # Class Function - drag_force
    # The Object_Model class function drag_force takes in arguments for:
    #     air_density - the density of air, set to default as 1.204 kg/m^3, the approximate air density at 101.325 kPa and 20 deg 
    #                    Celsius retrieved from en.wikipedia.org/wiki/Density_of_air
    #     verbose     - a boolean value that, if true, prints some debugging information into the console
    # The function calculates the drag force experienced by the object as a function of velocity from the equation for drag posted
    # by NASA at the website https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/drag-equation/
    # The function returns a 2D vector where the first index is the drag force in the x-direction, and the second is the drag-force
    # in the y-direction. It is important to note that the drag force always opposes the velocity of the object
    
    def drag_force(self, air_density = 1.204, verbose = False):
        if verbose:
            print(f"Velocity (x): {self.velocity_array[-1][0]} m/s")
            print(f"Velocity (y): {self.velocity_array[-1][1]} m/s")
            print(f"Drag force (x): {-1 * self.drag_coeff * air_density * 1/2 * self.velocity_array[-1][0] * np.abs(self.velocity_array[-1][0]) * self.cs_area}")
            print(f"Drag force (y): {-1 * self.drag_coeff * air_density * 1/2 * self.velocity_array[-1][1] * np.abs(self.velocity_array[-1][1]) * self.cs_area}")
        return [-1 * self.drag_coeff * air_density * 1/2 * self.velocity_array[-1][0] * np.abs(self.velocity_array[-1][0]) * self.cs_area, -1 * self.drag_coeff * air_density * 1/2 * self.velocity_array[-1][1] * np.abs(self.velocity_array[-1][1]) * self.cs_area]

    # Class Function - check_next_position
    # The Object_Model class function check_next_position takes in arguments for
    #     delta_time  - the time-step for Euler's method in seconds.
    #     air_density - the value for air density in kg/m^3
    #     gravity     - a boolean value that indicates whether gravitational acceleration is included in the model (True = gravity included)
    #     drag        - a boolean value that indicates whether drag force calculations are included in the model (True = drag included)
    #     verbose     - a boolean value that, if true, prints some debugging information into the console
    # The function calls another Object_Model class function, check_next_velocity, and it returns a 2D vector representing the next
    # position vector of the object. This is used to determine if the next iteration of Euler's method produces a collision or not.
    
    def check_next_position(self, delta_time=0.1, air_density = 1.204, gravity = False, drag = False, verbose = False):
        next_velocity = self.check_next_velocity(delta_time, air_density, gravity, drag)
        return [self.position_array[-1][0] + next_velocity[0] * delta_time, self.position_array[-1][1] + next_velocity[1] * delta_time]

    # Class Function - check_next_velocity
    # The Object_Model class function check_next_velocity takes in parameter arguments for
    #     delta_time  - the time-step for Euler's method in seconds.
    #     air_density - the value for air density in kg/m^3
    #     gravity     - a boolean value that indicates whether gravitational acceleration is included in the model (True = gravity included)
    #     drag        - a boolean value that indicates whether drag force calculations are included in the model (True = drag included)
    #     verbose     - a boolean value that, if true, prints some debugging information into the console
    # The function processes whether gravity and drag is included and depending on the input parameters, will return a 2D vector representing
    # the next velocity vector of the object using Euler's method. 
    # If only gravity is included: 
    #                       vx_next = vx                             vy_next = vy + g*dt
    # and if only drag is included:
    #                       vx_next = vx + Fdx/mass*dt               vy_next = vy + Fdy/mass*dt  
    # however if drag and gravity are included: 
    #                       vx_next = vx + Fdx/mass*dt               vy_next = vy + (g + Fdy/mass)*dt
    
    def check_next_velocity(self, delta_time=0.1, air_density = 1.204, gravity = False, drag = False, verbose = False):
        if not gravity and not drag:
            if verbose:
                print(f"Gravity: off\nDrag Force: off")
            return self.velocity_array[-1]
        elif gravity and not drag:
            if verbose:
                print(f"Gravity: on\nDrag Force: off")
            return [self.velocity_array[-1][0], self.velocity_array[-1][1] + (self.model_gravity * delta_time)]
        elif not gravity and drag:
            if verbose:
                print("Gravity: off\nDrag Force: on")
            return [self.velocity_array[-1][0] + (self.drag_force(air_density)[0] / self.mass * delta_time), self.velocity_array[-1][1] + (self.drag_force(air_density)[1] / self.mass) * delta_time]
        else:
            if verbose:
                print("Gravity: on\nDrag Force: on")
            return [self.velocity_array[-1][0] + (self.drag_force(air_density, verbose)[0] / self.mass * delta_time), self.velocity_array[-1][1] + (self.model_gravity + self.drag_force(air_density, verbose)[1] / self.mass) * delta_time]

    # Class Function - iterate_position
    # The Object_Model class function iterate_position takes in parameter arguments for
    #     delta_time  - the time-step for Euler's method in seconds.
    #     air_density - the value for air density in kg/m^3
    #     gravity     - a boolean value that indicates whether gravitational acceleration is included in the model (True = gravity included)
    #     drag        - a boolean value that indicates whether drag force calculations are included in the model (True = drag included)
    #     verbose     - a boolean value that, if true, prints some debugging information into the console
    # Within the function are some checks and balances against potential errors that may occur due to the approximations used in Euler's method
    # The function checks to determine if the object (inclusive of radius) is making contact with the ground and if the the vertical velocity is zero
    # and if so, then the position is reset such that the object is resting on the ground and the vertical velocity component remains zero.
    # The horizontal position in that case is updated according to whether drag is included.
    # The function then checks to determine if the position of the object is intersecting with the ground, and if the current velocity is less than zero
    # and in that case it resets the object to rest on the ground and fixes the vertical velocity to zero (collision handling is done within the model
    # itself and should update the velocity upon collision to be positive, this deals with the case where the model overshoots and puts the object underground)
    # Finally, outside of those cases, the object calculates the next position and velocity vectors according to the model settings.
    # In each of these cases the objects position_array and velocity_array is appended with the respective 2D vector, and the object's total_energy
    # vector is updated with the next iteration's kinetic and potential energies.
    
    def iterate_position(self, delta_time=0.1, air_density = 1.204, gravity = False, drag = False, verbose = False):
        if self.position_array[-1][1] - self.radius <= 0 and self.velocity_array[-1][1] == 0:
            if not drag:
                self.velocity_array.append([self.velocity_array[-1][0], self.velocity_array[-1][1]])
                self.position_array.append([self.position_array[-1][0] + self.velocity_array[-1][0] * delta_time, self.radius + self.velocity_array[-1][1] * delta_time])
            else:
                self.velocity_array.append([self.velocity_array[-1][0] + (self.drag_force(air_density)[0] / self.mass * delta_time), self.velocity_array[-1][1]])
                self.position_array.append([self.position_array[-1][0] + self.velocity_array[-1][0] * delta_time, self.radius + self.velocity_array[-1][1] * delta_time])
        elif self.check_next_position(delta_time,air_density, gravity, drag, False)[1] - self.radius <= 0 and self.check_next_velocity(delta_time, air_density, gravity, drag, False)[1] < 0:
            if not drag:
                self.velocity_array.append([self.velocity_array[-1][0], 0])
                self.position_array.append([self.position_array[-1][0] + self.velocity_array[-1][0] * delta_time, self.radius])
            else:
                self.velocity_array.append([self.velocity_array[-1][0] + (self.drag_force(air_density)[0] / self.mass * delta_time), 0])
                self.position_array.append([self.position_array[-1][0] + self.velocity_array[-1][0] * delta_time, self.radius])
        else:
            if not gravity and not drag:
                if verbose:
                    print(f"Gravity: off\nDrag Force: off")
                self.velocity_array.append([self.velocity_array[-1][0], self.velocity_array[-1][1]])
                self.position_array.append([self.position_array[-1][0] + self.velocity_array[-1][0] * delta_time, self.position_array[-1][1] + self.velocity_array[-1][1] * delta_time])
            elif gravity and not drag:
                if verbose:
                    print(f"Gravity: on\nDrag Force: off")
                self.velocity_array.append([self.velocity_array[-1][0], self.velocity_array[-1][1] + (self.model_gravity * delta_time)])
                self.position_array.append([self.position_array[-1][0] + self.velocity_array[-1][0] * delta_time, self.position_array[-1][1] + self.velocity_array[-1][1] * delta_time])
            elif not gravity and drag:
                if verbose:
                    print("Gravity: off\nDrag Force: on")
                self.velocity_array.append([self.velocity_array[-1][0] + (self.drag_force(air_density)[0] / self.mass * delta_time), self.velocity_array[-1][1] + (self.drag_force(air_density)[1] / self.mass) * delta_time])
                self.position_array.append([self.position_array[-1][0] + self.velocity_array[-1][0] * delta_time, self.position_array[-1][1] + self.velocity_array[-1][1] * delta_time])
            else:
                if verbose:
                    print("Gravity: on\nDrag Force: on")
                self.velocity_array.append([self.velocity_array[-1][0] + (self.drag_force(air_density)[0] / self.mass * delta_time), self.velocity_array[-1][1] + (self.model_gravity + self.drag_force(air_density, verbose)[1] / self.mass) * delta_time])
                self.position_array.append([self.position_array[-1][0] + self.velocity_array[-1][0] * delta_time, self.position_array[-1][1] + self.velocity_array[-1][1] * delta_time])
        self.total_energy.append([(1/2) * self.mass * np.linalg.norm(self.velocity_array[-1])**2, self.mass * np.abs(self.model_gravity) * self.position_array[-1][1]])

    # Class Function - print
    # The class function prints out critical information as to the object's mass, position vector, velocity vector, and radius.
    def print(self):
        print(
            f"mass = {self.mass:5.5f} kg, position = {self.position_array[-1]} m, velocity = {self.velocity_array[-1]} m/s, radius = {self.radius:5.2f} m")


# Class System_Model
# This class constructs a system in which the class Object_model is able to operate. The system takes the following input parameters in it's
# construction:
#     bottom_barrier     - a value for the bottom barrier of the model, currently set to y=0. This can be changed in the model however
#                           some code refactoring would be required such that the Object_Model would need to intake this value for its
#                           handling of Euler's method iterations. There is potential here for extensibility however it was unneeded for
#                           the scope of this project.
#     objects            - an array of Object_Model's with a size of at least 1. This project utilized Impulse-momentum theorem to 
#                           impulse and approximate force due to inelastic collisions between one moving and one immovable object,
#                           however the code framework for collisions between two objects is completed.
#     system_gravity     - a value for the system's acceleration due to gravity, currently set for -9.81 m/s^2
#     system_air_density - a value input for the system's air density, set to 1.204 kg/m^3
#     delta_time         - a value for the time-step of Euler's method in s
#     gravity_on         - a boolean value that, if true, allows for the implementation of gravity within the model
#     drag_force_on      - a boolean value that, if true, allows for the implementation of drag force within the model
#     window_angle       - a value for window angle (in degrees) for the angle of the normal vector for the window.
#                           the init function then takes the angle, converts to radians, and creates a unit normal vector
#                           for the window where n = <cos(window_angle), sin(window_angle) such that |n| = 1.
# The System_Model iterates itself according to input parameters given by the user, and calls on the system's Object_Model
# to iterate themselves

class System_Model:
    def __init__(self, bottom_barrier=0, objects=[Object_Model()], dt = 0.1, gravity_acceleration = -9.81, air_density = 1.204, gravity = False, drag = False, window_angle = 90):
        self.bottom_barrier = bottom_barrier
        self.objects = objects
        self.system_gravity = gravity_acceleration
        self.system_air_density = air_density
        self.delta_time = dt
        self.window_angle = np.radians(window_angle)
        self.window_normal_vector = np.array([np.cos(self.window_angle), np.sin(self.window_angle)])
        self.gravity_on = gravity
        self.drag_force_on = drag


    # Class Function - iterate_model
    # The System_Model class function iterate_model takes in parameters for:
    #     coeff_restitution - a value for the coefficient of restitution, e, to be used for the object within the system. As this project
    #                          simulates a single object colliding with an immovable surface, the coefficient of restitution is currently
    #                          input as a float, however it is easily extensible to allowing for an array of coefficients for multiple objects
    #                          if needed
    #     number_bounces    - a value for the number of bounces for which the system will iterate. As the research question is related to 
    #                          whether an object causes enough force due to impulse to break glass as a function of number of bounces before
    #                          contact, this value can be changed to examine different situations without having the model iterate beyond what's
    #                          needed.
    #     verbose           - a boolean value that, if true, prints some debugging information into the console
    # This function calls on several other class functions, namely collision_check() to check whether objects experience collisions
    # during each iteration of the model, and following collisions it calls the class function collision() to handle the collisions.
    
    def iterate_model(self, coeff_restitution, number_bounces = 1, verbose = False):
        iter = 0
        number_collisions = 0
        if verbose and iter == 0:
            print(f"\nIteration number {iter}\n")
            self.print()
        while number_collisions < number_bounces:
            number_collisions += 1
            while not self.collision_check()[0]:
                iter += 1
                for i, x in enumerate(self.objects):
                    x.iterate_position(self.delta_time, self.system_air_density, self.gravity_on, self.drag_force_on, verbose)
                if verbose:
                    print(f"\nIteration number {iter}")
                    self.print()
            if verbose:
                print(f"\nCollision number {number_collisions} during iteration number {iter}")
            self.collision(self.collision_check(verbose)[1], coeff_restitution, verbose)

    # The class function collision_check takes in a parameter for the change in time and checks the models objects sequentially to determine
    #     1) A collision between the object and the bottom edge
    #     2) A collision between two objects
    # The function returns a tuple where the first item returned is a boolean value for whether a collision occurs, and the second is a tuple [i, j]
    # returning the indices of the objects involved in an object-to-object collision. A collision between object index = i and any other objects yields results
    # in a j-value as follows:
    #     j = -1 : collision with the bottom barrier
    #     j != -1 : collision between object[i] and object[j] where j > i

    def collision_check(self, verbose = False):
        collisions_bool = False
        collisions = []
        if verbose:
            print(f"\nChecking collisions:")
        for i, obj in enumerate(self.objects):
            obj_next_position = obj.check_next_position(self.delta_time, self.system_air_density, self.gravity_on, self.drag_force_on, verbose)
            if verbose:
                print(f"Object next position = ({obj_next_position[0]},{obj_next_position[1]})")
            if obj_next_position[1] - obj.radius < self.bottom_barrier:
                if verbose:
                    print(f"Collision between object {i} and bottom barrier")
                collisions_bool = True
                collisions.append((i, -1))
            for j, second_obj in enumerate(self.objects[i + 1:]):
                second_obj_next_position = second_obj.check_next_position(self.delta_time, self.system_air_density, self.gravity_on, self.drag_force_on, verbose)
                distance = np.sqrt((obj_next_position[0] - second_obj_next_position[0])**2 + (obj_next_position[1] - second_obj_next_position[1])**2)
                if verbose:
                    print(f"Distance between object {i} and object {i + j + 1} is {distance}")
                if distance < obj.radius + second_obj.radius:
                    if verbose:
                        print(f"\nCollision between object {i} and object {j+i+1}")
                        print(f"Distance {distance} is less than {obj.radius + second_obj.radius}")
                    collisions_bool = True
                    collisions.append((i, i+j+1))
        return collisions_bool, collisions

    # Collision handling where distance between objects is less than the time step delta_time are simplified
    # such that because the time before impact is less than the time step, the calculations for time remaining
    # before impact take the last velocity as static and does not update that velocity with respect to gravity
    # or drag.
    
    # Class Function - collision
    # The class function collision takes in the following parameters:
    #     collisions        - a 2D array where the first index represents the index of the System_Model's objects array for the first object
    #                          involved in the collision, and the second index represents the index of the second object (or -1 for the bottom
    #                          barrier.
    #     coeff_restitution - a value for the coefficient of restitution, e, to be used for the object within the system.
    #     verbose           - a boolean value that, if true, prints some debugging information into the console
    # This function handles collisions between an object and a barrier, which is the focus of this project, and as well has code in 
    # order to calculate collisions between to objects. The class uses a coefficient of restitution in order to calibrate for either
    # elastic or inelastic collisions, and the function updates the position, velocity, impulse and total_energy arrays of each object
    # within the model within each iteration. The velocity array for the object is updated as though the ball has hit the ground
    # in order to continue simulating the next bounce uninterrupted, however the force due to impulse array on each bounce is updated
    # where the change in momentum uses the object mass and the dot product between each of the pre-/post-collision velocities and the
    # unit normal vector for the window, or the scalar projection of the magnitude of each velocity onto the unit normal vector for the window.

    def collision(self, collisions = [], coeff_restitution = 1, verbose = False):
        time_after_collision = 0
        time_before_collision = 0
        impulse = 0
        if verbose:
            print(f"\nEntering collision handling")
            print(f"Collision code: {collisions}")
        match collisions[0][1]:
            case -1:
                current_velocity = np.array(self.objects[collisions[0][0]].velocity_array[-1])
                next_velocity = self.objects[collisions[0][0]].check_next_velocity(self.delta_time, self.system_air_density, self.gravity_on, self.drag_force_on)
                time_before_collision = (self.bottom_barrier - self.objects[collisions[0][0]].position_array[-1][1])/current_velocity[1]
                time_after_collision = np.abs(self.delta_time - time_before_collision)
                position_at_collision = []
                position_after_collision = []
                
                if verbose:
                    print(f"Collision handling for collision between object {collisions[0][0]} and bottom barrier")
                    print(f"Time before collision: {time_before_collision}")
                    print(f"Time after collision: {time_after_collision} s")
                    self.objects[collisions[0][0]].print()
                    print(f"Object {collisions[0][0]} hit the ground with impact velocity {self.objects[collisions[0][0]].velocity_array[-1][1]} m/s")
                
                position_at_collision = [self.objects[collisions[0][0]].position_array[-1][0] + next_velocity[0] * time_before_collision, self.objects[collisions[0][0]].radius]
                
                self.objects[collisions[0][0]].velocity_array.append([self.objects[collisions[0][0]].velocity_array[-1][0], -coeff_restitution * next_velocity[1]])
                
                position_after_collision = [position_at_collision[0] + self.objects[collisions[0][0]].velocity_array[-1][0] * time_after_collision, position_at_collision[1] + self.objects[collisions[0][0]].velocity_array[-1][1] * time_after_collision]
                
                self.objects[collisions[0][0]].position_array.append(position_after_collision)
                self.objects[collisions[0][0]].total_energy.append([(1/2) * self.objects[collisions[0][0]].mass * np.linalg.norm(self.objects[collisions[0][0]].velocity_array[-1])**2, self.objects[collisions[0][0]].mass * np.abs(self.objects[collisions[0][0]].model_gravity) * self.objects[collisions[0][0]].position_array[-1][1]])
                
                if verbose:
                    print(f"Object {collisions[0][0]} left the ground with velocity {self.objects[collisions[0][0]].velocity_array[-1][1]} m/s")
                    print(f"Y position: {self.objects[collisions[0][0]].position_array[-1][1]}\nY velocity: {self.objects[collisions[0][0]].velocity_array[-1][1]}")
                    print(f"Delta V = {np.subtract(self.objects[collisions[0][0]].velocity_array[-1], current_velocity)} m/s")
                    print(f"m * Delta V = {self.objects[collisions[0][0]].mass * np.subtract(self.objects[collisions[0][0]].velocity_array[-1], current_velocity)} kgm/s")
                    print(f"m * Delta V scalar = {self.objects[collisions[0][0]].mass * np.linalg.norm(np.subtract(self.objects[collisions[0][0]].velocity_array[-1], current_velocity))} kgm/s")
                    print(f"Force due to Impulse = {self.objects[collisions[0][0]].mass * np.linalg.norm(np.subtract(self.objects[collisions[0][0]].velocity_array[-1], current_velocity)) / self.delta_time} N")
                
                v_current_norm = np.dot(current_velocity, self.window_normal_vector)
                v_next_norm = -coeff_restitution*v_current_norm
                #v_next_norm = np.dot(-coeff_restitution*current_velocity, self.window_normal_vector)
                impulse = self.objects[collisions[0][0]].mass*(v_next_norm - v_current_norm)/self.delta_time
                self.objects[collisions[0][0]].force_due_to_impulse_array.append(impulse)
        
            case _:
                if verbose:
                    print(f"Collision handling for collision between object {collisions[0][0]} and object {collisions[0][1]}")
                x1_last = np.array(self.objects[collisions[0][0]].position_array[-1])
                x2_last = np.array(self.objects[collisions[0][1]].position_array[-1])
                v1_last = np.array(self.objects[collisions[0][0]].velocity_array[-1])
                v2_last = np.array(self.objects[collisions[0][1]].velocity_array[-1])

                delta_x = x1_last - x2_last
                delta_v1 = v1_last - v2_last
                delta_v2 = v2_last - v1_last

                distance_squared = np.inner(delta_x, delta_x)
                
                init_momentum = np.linalg.norm(m1 * v1_last + m2 * v2_last)

                v1_final = ((m1 - e * m2) / (m1 + m2)) * v1_last + ((1 + e) * m2 / (m1 + m2)) * v2_last
                v2_final = ((m2 - e * m1) / (m1 + m2)) * v2_last + ((1 + e) * m1 / (m1 + m2)) * v1_last

                self.objects[collisions[0][0]].velocity_array.append(v1_final)
                self.objects[collisions[0][0]].position_array.append([self.objects[collisions[0][0]].position_array[-1][0] + self.objects[collisions[0][0]].velocity_array[-1][0]*self.delta_time, self.objects[collisions[0][0]].position_array[-1][1] + self.objects[collisions[0][0]].velocity_array[-1][1]*self.delta_time])
                self.objects[collisions[0][0]].total_energy.append([(1/2) * self.objects[collisions[0][0]].mass * np.linalg.norm(self.objects[collisions[0][0]].velocity_array[-1])**2, self.objects[collisions[0][0]].mass * np.abs(self.objects[collisions[0][0]].model_gravity) * self.objects[collisions[0][0]].position_array[-1][1]])
                self.objects[collisions[0][0]].force_due_to_impulse_array.append(self.objects[collisions[0][0]].mass * np.linalg.norm(np.subtract(self.objects[collisions[0][0]].velocity_array[-1], self.objects[collisions[0][0]].velocity_array[-2])) / self.delta_time)
                
                self.objects[collisions[0][1]].velocity_array.append(v2_final)
                self.objects[collisions[0][1]].position_array.append([self.objects[collisions[0][1]].position_array[-1][0] + self.objects[collisions[0][1]].velocity_array[-1][0]*self.delta_time, self.objects[collisions[0][1]].position_array[-1][1] + self.objects[collisions[0][1]].velocity_array[-1][1]*self.delta_time])
                self.objects[collisions[0][1]].total_energy.append([(1/2) * self.objects[collisions[0][1]].mass * np.linalg.norm(self.objects[collisions[0][1]].velocity_array[-1])**2, self.objects[collisions[0][1]].mass * np.abs(self.objects[collisions[0][0]].model_gravity) * self.objects[collisions[0][0]].position_array[-1][1]])
                self.objects[collisions[0][1]].force_due_to_impulse_array.append(self.objects[collisions[0][1]].mass * np.linalg.norm(np.subtract(self.objects[collisions[0][1]].velocity_array[-1], self.objects[collisions[0][1]].velocity_array[-2])) / self.delta_time)
    
    
    # Class Function - print
    # The System_Model class function print() iterates through the Object_Model array of the system and calls the Object_Model function
    # print() for each object within the array.
    
    def print(self):
        for i,x in enumerate(self.objects):
            print(f"Object {i+1}:")
            x.print()

#########################################################################################################################################

## Some constant values for the system model

# Baseball dimensions retrieved from img.mlbstatic.com/mlb-images/image/upload/mlb/atcjzj9j7wrgvsm8wnjq.pdf pg. 5
# from section 3.00 - EQUIPMENT AND UNIFORMS - Rule 3.0.1 - The Ball
baseball_mass = 0.14529131 # kg
baseball_radius = 0.1158875 # m

# Approximate air density at 101.325 kPa and 20 deg Celsius retrieved from en.wikipedia.org/wiki/Density_of_air
air_density = 1.204 # kg/m^3

# Conversion rate from MPH to m/s
mph_conversion = 1/2.237

# Statistics relating to home-run (HR) in the MLB.

# Exit velocities (EV) that yield an HR percent greater that 20% in MPH
# collected from baseballsavant.mlb.com/statcast_hit_probability?year=2024&type=ev

exit_velocities_mph = [106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120] # mph

# EV conversion from MPH to m/s
exit_velocities = [mph_conversion * v for v in exit_velocities_mph]

# Average EV in m/s
exit_velocity_average = np.sum(exit_velocities)/len(exit_velocities)

# Launch angles (LA) that yield a HR percent greater than 20% in degrees
# collected from baseballsavant.mlb.com/statcast_hit_probability?year=2024&type=la 
launch_angles_deg = [24, 25, 26, 27, 28, 29, 30, 31, 32]

# Average Launch angles (LA) in degrees
launch_angles_average_deg = np.sum(launch_angles_deg)/len(launch_angles_deg)

# Average launch angles (LA) in radian
launch_angle_average = launch_angles_average_deg * np.pi/180

# Exit velocities as 2D velocity vectors
exit_vectors = [exit_velocity_average*np.cos(launch_angle_average), exit_velocity_average * np.sin(launch_angle_average)]

# Time step for Euler's Method
dt = 0.001 #s

# Acceleration due to gravity, g
g = -9.81 # m/s^2

# Force required to break a windshield
newton_force = 2100 # N

# Boolean value that, if true, will cause the code in the Graphs of Motion cells to execute.
# As the graphs of motion were not the focus of the project, but rather a qualitative analysis tool to verify expected model behaviour,
# the graphs are by default NOT printed for each model examined AFTER system test where the collision was inelastic and the the total
# energy conserved
#
# If you would like to verify that the Graphs of Motion are as described in the report, please change the below value to True and 
# re-run the notebook

# Boolean flags for Graphs of Motion / Verbose outputs. 
graphs_motion_bool = False
verbose = False