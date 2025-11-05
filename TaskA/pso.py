# Real-valued PSO with boundary handling (param ranges) and parallelized cost function evaluation
# The optimize() function contains the whole optimization loop

import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time
# Import logger to ensure global print override is active
import logger

class PSO:
    def __init__(self, cost_func, bounds, num_particles=50, max_iter=200, w=0.7, c1=1.5, c2=1.5, max_workers=None):
        self.original_cost_func = cost_func
        self.original_bounds = np.array(bounds)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.dim = len(bounds)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_workers = max_workers
        
        # Calculate normalization parameters
        self.param_mins = self.original_bounds[:, 0]
        self.param_ranges = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        
        # PSO will work in normalized [0,1] space
        self.normalized_bounds = np.array([[0.0, 1.0] for _ in range(self.dim)])
    
    def normalize_params(self, params):
        """Convert from original parameter space to [0,1] normalized space."""
        return (params - self.param_mins) / self.param_ranges
    
    def denormalize_params(self, normalized_params):
        """Convert from [0,1] normalized space back to original parameter space."""
        return normalized_params * self.param_ranges + self.param_mins
    
    def normalized_cost_function(self, normalized_params):
        """Wrapper that denormalizes params before calling original cost function."""
        original_params = self.denormalize_params(normalized_params)
        return self.original_cost_func(original_params)


    def optimize(self):
        start_time = time.time()
        
        # Log parallel processing configuration
        print(f"Starting PSO optimization...")
        print(f"Parameters: {self.num_particles} particles, {self.max_iter} iterations")
        print(f"Using parallel evaluation with {self.max_workers} workers")
        print(f"Working in normalized [0,1] parameter space")
        
        # Initialize particles and velocities in normalized [0,1] space
        particles = np.random.uniform(0.0, 1.0, (self.num_particles, self.dim))
        velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, self.dim))  # Small initial velocities
        personal_best = particles.copy()
        
        # Track min/max explored values in ORIGINAL parameter space
        original_particles = np.array([self.denormalize_params(p) for p in particles])
        min_explored = np.min(original_particles, axis=0).copy()
        max_explored = np.max(original_particles, axis=0).copy()
        
        # Evaluate initial scores (sequential if max_workers <= 1 to avoid multiprocessing issues)
        if self.max_workers is not None and self.max_workers <= 1:
            personal_best_scores = np.array([self.normalized_cost_function(p) for p in particles])
        else:
            # Parallel evaluation of initial scores using normalized cost function
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                personal_best_scores = np.array(list(executor.map(self.normalized_cost_function, particles)))
        
        global_best = personal_best[np.argmin(personal_best_scores)].copy()
        global_best_score = np.min(personal_best_scores)

        iteration_times = []

        for i in range(self.max_iter):
            iter_start_time = time.time()
            
            # Update particles and velocities in normalized space
            for j in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[j] = (self.w * velocities[j] +
                                 self.c1 * r1 * (personal_best[j] - particles[j]) +
                                 self.c2 * r2 * (global_best - particles[j]))
                particles[j] += velocities[j]
                # Clip to normalized bounds [0,1]
                particles[j] = np.clip(particles[j], 0.0, 1.0)
            
            # Track min/max explored values in original parameter space
            original_particles = np.array([self.denormalize_params(p) for p in particles])
            min_explored = np.minimum(min_explored, np.min(original_particles, axis=0))
            max_explored = np.maximum(max_explored, np.max(original_particles, axis=0))
            
            # Evaluate scores (sequential if max_workers <= 1)
            if self.max_workers is not None and self.max_workers <= 1:
                scores = np.array([self.normalized_cost_function(p) for p in particles])
            else:
                # Parallel evaluation of scores for all particles using normalized cost function
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    scores = np.array(list(executor.map(self.normalized_cost_function, particles)))
            
            # Update personal bests
            for j in range(self.num_particles):
                score = scores[j]
                if score < personal_best_scores[j]:
                    personal_best[j] = particles[j].copy()
                    personal_best_scores[j] = score
            
            # Update global best
            best_idx = np.argmin(personal_best_scores)
            if personal_best_scores[best_idx] < global_best_score:
                global_best = personal_best[best_idx].copy()
                global_best_score = personal_best_scores[best_idx]
            
            iter_end_time = time.time()
            iter_duration = iter_end_time - iter_start_time
            iteration_times.append(iter_duration)
            
            print(f"Iter {i+1}/{self.max_iter}, Best Loss: {global_best_score:.6f}")
        
        total_time = time.time() - start_time
        mean_time_per_iter = np.mean(iteration_times)
        
        # Denormalize the best solution before returning
        global_best_original = self.denormalize_params(global_best)
        
        return global_best_original, global_best_score, min_explored, max_explored, total_time, mean_time_per_iter
