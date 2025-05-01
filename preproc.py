import numpy as np
import scipy.ndimage.filters as filters
import torch
import skimage as sk
import random
import torch.nn.functional as F
import random
import numpy as np


class Transpose(object):
    def __init__(self):
        pass
    def __call__(self, data):
        return data.transpose(-1,-2)

    
class Gray(object):
    def __init__(self):
        pass
    def __call__(self, data):
        return data.mean(-3, keepdim=True)
    
    
class PermutationNoise(object):
    def __init__(self):
        pass
    def __call__(self, data):
        shape = data.shape
        new_data = 0*data # Creates a tensor of the same shape as data
        idx = [torch.tensor(np.random.permutation(np.prod(shape[-2:])))] # Generate Permutation Indices: Flattens the last two dimensions (H × W) into a single dimension and generates a random permutation of indices for all pixels.
        for i, x in enumerate(data): # Permute Each Channel:
            new_data[i] = (x.view(np.prod(shape[-2:]))[idx]).view(shape[-2:])
        return new_data

    
class GaussianFilter(object):
    def __init__(self):
        pass

    def __call__(self, data):
        sigma = 1. + 1.5 * torch.rand(1).item()
        return torch.tensor(filters.gaussian_filter(data, sigma, mode='reflect'))
    
    
class ContrastRescaling(object):
    def __init__(self):
        pass

    def __call__(self, data):
        gamma = 5 + 25. * torch.rand(1).item()
        return torch.sigmoid(gamma*(data-.5))
    
    
class AddSpeckleNoise(object):
    def __init__(self, mean=0.1, sd=0.1):
        """
        Args:
            mean (float): Mean of the speckle noise distribution.
            sd (float): Standard deviation of the speckle noise distribution.
        """
        self.mean = mean
        self.sd = sd

    def __call__(self, sample):
        """
        Args:
            sample (Tensor): Input image tensor.
        Returns:
            Tensor: Image tensor with speckle noise added.
        """
        noise = torch.randn(sample.size()) * self.sd + self.mean
        noisy_sample = sample + sample * noise
        return noisy_sample


class Fundus_impulse_noise(object):  
    def __init__(self, noise_lvl=1):
        self.noise_lvl = noise_lvl
        self.val=0
    def __call__(self, x):
        if self.noise_lvl == 'minimal':
            severity = 0.01 
        elif self.noise_lvl == 'low':
            severity = 0.05 #random.uniform(0,0.1) 
        elif self.noise_lvl == 'moderate':
            severity = 1 #random.uniform(0.06,0.8)
        elif self.noise_lvl == 'high':        
            severity = 2
        self.val = severity
        x = sk.util.random_noise(np.array(x), mode='s&p', amount=severity)
        return np.clip(x, 0, 1) 
    def __repr__(self):
            return f"Fundus_impulse_noise(val={self.val:.2f})"

class Fundus_AddSpeckleNoise(object):
    def __init__(self, noise_lvl):
        self.noise_lvl = noise_lvl
        self.val=0
    def __call__(self, sample):
        if self.noise_lvl == 'low':
            sd = 0.05 #random.uniform(0,0.1)  # 
        elif self.noise_lvl == 'mid':
            sd = 1 #random.uniform(0.06,0.8)
        elif self.noise_lvl == 'high':
            sd = 1 #random.uniform(1,1.5)
        self.val = sd

        noise = torch.randn(sample.size()) * sd + 0.1
        noisy_sample = sample + sample * noise
        return noisy_sample
    def __repr__(self):
            return f"Fundus_AddSpeckleNoise(mean={0.5},sd={self.val:.2f})"


class Fundus_GaussianFilter(object):
    def __init__(self, noise_lvl):
        self.noise_lvl = noise_lvl
        self.val=0
    def __call__(self, data):
        if self.noise_lvl == 'minimal':
            sigma = 0.4 # random.uniform(0.1,0.6)
        elif self.noise_lvl == 'low':
            sigma = 0.7 #random.uniform(1,1.5) # random.uniform(0.7,1)
        elif self.noise_lvl == 'moderate':
            sigma = 1.5 #random.uniform(2.5, 4.5) # random.uniform(1.2,2.5)      
        elif self.noise_lvl == 'high':
            sigma = 5 #random.uniform(2.5, 4.5) # random.uniform(1.2,2.5)        
        self.val = sigma
        return torch.tensor(filters.gaussian_filter(data, sigma, mode='reflect'))
    def __repr__(self):
            return f"Fundus_GaussianFilter(sigma={self.val:.2f})"


class Fundus_ContrastRescaling(object):
    def __init__(self, noise_lvl):
        self.noise_lvl = noise_lvl
        self.val=0
    def __call__(self, data):
        if self.noise_lvl == 'minimal':
            gamma = 6
        elif self.noise_lvl == 'low':
            gamma = 4 #random.uniform(10,3)
        elif self.noise_lvl == 'moderate':
            gamma = 2.5 #random.uniform(2,1)
        elif self.noise_lvl == 'high':
            gamma = 0.3 #random.uniform(0.6,0.05)    
        self.val=gamma
        return torch.sigmoid(gamma*(data-.5))
    def __repr__(self):
            return f"Fundus_ContrastRescaling(gamma={self.val:.2f})"


class Fundus_PermutationNoise(object):
    def __init__(self, noise_lvl):
        self.noise_lvl = noise_lvl
        self.val=0
    def __call__(self, data):
        shape = data.shape
        new_data = data.clone()  # Make a copy of the original data
        num_pixels = np.prod(shape[-2:])
        if self.noise_lvl == 'minimal':
            max_permutation_size = 0.04
        elif self.noise_lvl == 'low':
            max_permutation_size = 0.06  #random.uniform(0.007,0.02)
        elif self.noise_lvl == 'moderate':
            max_permutation_size = 0.1 #random.uniform(0.04,0.07)
        elif self.noise_lvl == 'high':
            max_permutation_size = 0.7 #random.uniform(0.1,0.3)   


        # Determine the maximum number of pixels to shuffle based on max_permutation_size
        max_pixels_to_shuffle = int(max_permutation_size * num_pixels)
        self.val = max_permutation_size

        # Randomly select a subset of pixels to shuffle
        # Set the seed for numpy to ensure the randomness is fixed
        idx = np.random.choice(num_pixels, size=max_pixels_to_shuffle, replace=False)

        for i in range(len(data)):  # Loop through each channel
            # Flatten the data and shuffle only the selected pixels
            flat_data = data[i].view(-1)
            # flat_data[idx] = torch.rand_like(flat_data[idx])  # Shuffle by assigning random values
            flat_data[idx] = flat_data[torch.randperm(len(idx))]


            # Reshape the shuffled data back to its original shape
            new_data[i] = flat_data.view(shape[-2:])
        return new_data  
    def __repr__(self):
            return f"Fundus_PermutationNoise(permt_pixels={self.val:.2f})"

class Fundus_SaltPepperNoise(object):
    def __init__(self, noise_lvl):
        self.noise_lvl = noise_lvl
        self.val = 0
    
    def __call__(self, data):
        # Convert the image to numpy for easier manipulation (if needed)
        data = data.numpy()
        
        # Define the noise level in terms of percentage of pixels to corrupt
        if self.noise_lvl == 'minimal':
            salt_pepper_ratio = 0.001  # 1% noise
        elif self.noise_lvl == 'low':
            salt_pepper_ratio = 0.005  # 5% noise
        elif self.noise_lvl == 'moderate':
            salt_pepper_ratio = 0.01  # 10% noise
        elif self.noise_lvl == 'high':
            salt_pepper_ratio = 0.1  # 20% noise
        
        self.val = salt_pepper_ratio

        # Get the total number of pixels
        num_pixels = data.size

        # Calculate how many pixels should be corrupted
        num_noise_pixels = int(salt_pepper_ratio * num_pixels)

        # Add salt-and-pepper noise
        for _ in range(num_noise_pixels):
            # Randomly select a pixel
            i = random.randint(0, data.shape[0] - 1)
            j = random.randint(0, data.shape[1] - 1)
            
            # Randomly assign it to salt (white) or pepper (black)
            if random.random() < 0.5:
                data[i, j] = 1  # Salt (white)
            else:
                data[i, j] = 0  # Pepper (black)

        # Convert back to tensor
        noisy_data = torch.tensor(data)

        return noisy_data

    def __repr__(self):
        return f"Fundus_SaltPepperNoise(noise_lvl={self.val:.2f})"


# ==============================# ===============================
# =============== For test and inference analysis ===============
# ==============================# ===============================
class Fundus_GaussianFilter_test(object):
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, data):
        sigma = self.severity 
        return torch.tensor(filters.gaussian_filter(data, sigma, mode='reflect'))

# class Fundus_GaussianFilter_test(object):
#     def __init__(self, severity=5):
#         self.severity = severity

#     def __call__(self, data):
#         # Ensure data is in the correct format (3D tensor for RGB images)
#         if len(data.shape) == 3 and data.shape[0] == 3:  # Assuming CxHxW format
#             # Apply Gaussian filter to each channel separately
#             filtered_data = torch.zeros_like(data)
#             for i in range(data.shape[0]):  # Loop over the RGB channels
#                 filtered_data[i] = torch.tensor(filters.gaussian_filter(data[i].numpy(), self.severity, mode='reflect'))
#             return filtered_data
#         else:
#             raise ValueError("Input data must be a 3D tensor representing a color image (C, H, W)")



class Fundus_ContrastRescaling_test(object):
    def __init__(self, severity=5):
        self.severity = severity

    def __call__(self, data):
        gamma = self.severity  # gamma = self.severity + 15 * torch.rand(1).item()
        return torch.sigmoid(gamma*(data-.5))


class AddSpeckleNoise_test(object):
    def __init__(self, mean=0.5, sd=0.1, seed=42):
        self.mean = mean
        self.sd = sd
        self.seed = seed

    def __call__(self, sample):
        # torch.manual_seed(self.seed)  # Fix the seed for reproducibility
        noise = torch.randn(sample.size()) * self.sd + self.mean
        noisy_sample = sample + sample * noise
        return noisy_sample
    
    
class Fundus_PermutationNoise_test(object):
    def __init__(self, max_permutation_size=0.1, seed=42):
        self.max_permutation_size = max_permutation_size
        self.seed = seed  # Add a seed parameter for reproducibility

    def __call__(self, data):
        torch.manual_seed(self.seed)  # Fix the seed for reproducibility
        shape = data.shape
        new_data = data.clone()  # Make a copy of the original data
        num_pixels = np.prod(shape[-2:])

        # Determine the maximum number of pixels to shuffle based on max_permutation_size
        max_pixels_to_shuffle = int(self.max_permutation_size * num_pixels)

        # Randomly select a subset of pixels to shuffle
        # Set the seed for numpy to ensure the randomness is fixed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed) # Fix the seed for numpy random operations
        idx = np.random.choice(num_pixels, size=max_pixels_to_shuffle, replace=False)

        for i in range(len(data)):  # Loop through each channel
            # Flatten the data and shuffle only the selected pixels
            flat_data = data[i].view(-1)
            # flat_data[idx] = torch.rand_like(flat_data[idx])  # Shuffle by assigning random values
            flat_data[idx] = flat_data[torch.randperm(len(idx))]


            # Reshape the shuffled data back to its original shape
            new_data[i] = flat_data.view(shape[-2:])
        return new_data  



# ==============================# ===============================
# ===============================================================
# ==============================# ===============================

    
class AdversarialNoise(object):
    def __init__(self, model, device, epsilon=0.3):
        self.model = model
        self.pretransform = dl.noise_transform
        self.device = device
        self.epsilon = epsilon
        
    def __call__(self, data):
        perturbed = tt.generate_adv_noise(self.model, self.epsilon, 
                                       device=self.device, batch_size=1, 
                                       norm=20, num_of_it=40, 
                                       alpha=0.01, seed_images=data.unsqueeze(0))
        return perturbed.squeeze(0)