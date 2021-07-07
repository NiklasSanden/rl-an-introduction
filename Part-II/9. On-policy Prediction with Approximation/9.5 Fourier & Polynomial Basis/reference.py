# Here is a previous unused, untested, and highly unoptimized implementation of Polynomial & Fourier Basis that used to sit
# in the function_approximators.py file. It can take multi-dimensional input. The dependencies might not be up-to-date,
# but I will leave it here as a potential reference for how it could be done. It gives all (n + 1) ^ k possibilites as features. 

class PolynomialBasis(GradientFunctionApproximator):
    '''
    Can take any dimensional input as a numpy array or a scalar.
    '''
    def __init__(self, n, k, highest_state_values):
        self.n = n
        self.d = (n + 1) ** k
        self.highest_state_values = highest_state_values
        super().__init__(self.d, np.zeros(self.d))

    def __call__(self, input):
        feature_vector = self._construct_feature_vector(input)
        return np.dot(feature_vector, self.weights)

    def get_gradients(self, input):
        return self._construct_feature_vector(input)
    
    def _construct_feature_vector(self, input):
        if np.isscalar(input):
            input = np.array([input])
        input = input / self.highest_state_values
        
        # Precalculate
        exps_of_s = np.ones((self.n + 1, len(input)))
        for i in range(1, self.n + 1):
            exps_of_s[i, :] = np.multiply(exps_of_s[i - 1, :], input)
        
        powers_of_n_plus_one = np.ones(len(input), dtype=int)
        for i in range(1, len(powers_of_n_plus_one)):
            powers_of_n_plus_one[i] = powers_of_n_plus_one[i - 1] * (self.n + 1)

        # Construct
        features = np.zeros(self.d)
        for i in range(self.d):
            indices = [((i // powers_of_n_plus_one[k]) % (self.n + 1)) for k in range(len(input))]
            features[i] = np.product([exps_of_s[indices[k], k] for k in range(len(input))])
        
        return features

class FourierBasis(GradientFunctionApproximator):
    '''
    Can take any dimensional input as a numpy array or a scalar.
    '''
    def __init__(self, n, k, highest_state_values):
        self.n = n
        self.d = (n + 1) ** k
        self.highest_state_values = highest_state_values
        super().__init__(self.d, np.zeros(self.d))

    def __call__(self, input):
        feature_vector = self._construct_feature_vector(input)
        return np.dot(feature_vector, self.weights)

    def get_gradients(self, input):
        return self._construct_feature_vector(input)
    
    def _construct_feature_vector(self, input):
        if np.isscalar(input):
            input = np.array([input])
        input = input / self.highest_state_values
        
        # Precalculate
        powers_of_n_plus_one = np.ones(len(input), dtype=int)
        for i in range(1, len(powers_of_n_plus_one)):
            powers_of_n_plus_one[i] = powers_of_n_plus_one[i - 1] * (self.n + 1)

        # Construct
        features = np.zeros(self.d)
        for i in range(self.d):
            constant_factors = [((i // powers_of_n_plus_one[k]) % (self.n + 1)) for k in range(len(input))]
            features[i] = np.cos(math.pi * np.dot(input, np.array([constant_factors[k] for k in range(len(input))])))
        
        return features