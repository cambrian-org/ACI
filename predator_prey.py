import numpy as np

class PredatorPreyEnvironment:
    def __init__(self, 
                 world_size=10,
                 prey_speed=0.5,
                 predator_speed=0.7,
                 prey_energy=100,
                 predator_energy=100,
                 energy_consumption=1):
        """
        Initialize the predator-prey environment with individual agents.
        
        Parameters:
        - world_size: Size of the square world
        - prey_speed: Movement speed of prey
        - predator_speed: Movement speed of predator
        - prey_energy: Initial energy of prey
        - predator_energy: Initial energy of predator
        - energy_consumption: Energy consumed per step
        """
        self.world_size = world_size
        self.prey_speed = prey_speed
        self.predator_speed = predator_speed
        self.energy_consumption = energy_consumption
        
        # Initialize positions and states
        self.reset()
        
    def reset(self):
        """Reset the environment with random positions"""
        # Random positions for prey and predator
        self.prey_pos = np.random.uniform(0, self.world_size, 2)
        self.predator_pos = np.random.uniform(0, self.world_size, 2)
        
        # Initialize energies
        self.prey_energy = 100
        self.predator_energy = 100
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state of the environment"""
        return {
            'prey_pos': self.prey_pos,
            'predator_pos': self.predator_pos,
            'prey_energy': self.prey_energy,
            'predator_energy': self.predator_energy
        }
    
    def step(self, prey_action=None, predator_action=None):
        """
        Perform one step of the simulation.
        
        Parameters:
        - prey_action: Direction vector for prey movement (optional)
        - predator_action: Direction vector for predator movement (optional)
        
        Returns:
        - state: Current state of the environment
        - done: Whether the episode is done
        - info: Additional information
        """
        # Default random movement if no action provided
        if prey_action is None:
            prey_action = np.random.uniform(-1, 1, 2)
        if predator_action is None:
            predator_action = np.random.uniform(-1, 1, 2)
        
        # Normalize action vectors
        prey_action = prey_action / (np.linalg.norm(prey_action) + 1e-8)
        predator_action = predator_action / (np.linalg.norm(predator_action) + 1e-8)
        
        # Update positions
        self.prey_pos += prey_action * self.prey_speed
        self.predator_pos += predator_action * self.predator_speed
        
        # Keep within world bounds
        self.prey_pos = np.clip(self.prey_pos, 0, self.world_size)
        self.predator_pos = np.clip(self.predator_pos, 0, self.world_size)
        
        # Update energies
        self.prey_energy -= self.energy_consumption
        self.predator_energy -= self.energy_consumption
        
        # Check if predator caught prey
        distance = np.linalg.norm(self.prey_pos - self.predator_pos)
        done = False
        if distance < 0.5:  # Catch radius
            done = True
            self.predator_energy += 50  # Energy gain from catching prey
        
        # Check if anyone ran out of energy
        if self.prey_energy <= 0 or self.predator_energy <= 0:
            done = True
        
        return self._get_state(), done, {'distance': distance} 