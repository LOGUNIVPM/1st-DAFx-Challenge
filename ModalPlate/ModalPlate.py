# ---------------------------------------------------------
# Modal Plate Class by Michele Ducceschi, University of Bologna
# Python Class Implementation by Matthew Hamilton.
#
# ---------------------------------------------------------

import numpy as np

# Import logger to ensure global print override is active
try:
    import logger
except ImportError:
    pass  # Logger module may not be available in all contexts

import os
import soundfile as sf
from scipy.io.wavfile import write
# ---------------------------------------------------------


class ModalPlate:
    """
    Order of events:
    Initialise:
        - ModalPlate.init()
    Re-initialise:
        - ModalPlate.populate_params()
        - ModalPlate.setup()
    Synthesis:
        - ModalPlate.update()


    TODO:
        - set some safeguards to make sure the plate is actually setup before attempting to use the update method
        - allow setup to be called multiple times
    """

    # ---------------------------------------------------------

    def __init__(self,
                 sample_rate: float = 44100,
                 plate_params: dict = None):
        """

        :param sample_rate: Internal audio sample rate
        :param plate_params: Dictionary of plate parameters
        """
        self.sample_rate = sample_rate
        self.fmax = 10000.0      # Largest modal frequency
        if plate_params is None:
            print("No plate_params supplied — using default ones!")
            self.plate_params = {
            'Lx': 0.5,
            'Ly': 1.1,
            'h': 0.001,
            'T0': 0.01,
            'rho': 2430.0,
            'E': 6.7e10,
            'nu': 0.25,
            'T60_DC': 6.0,
            'T60_F1': 1.0,
            'loss_F1': 500.0,
            'fp_x': 0.1,
            'fp_y': 0.1,
            'op_x': 0.61,
            'op_y': 0.61
            }
        else:
            self.plate_params = plate_params
        
        self.populate_params()
        self.setup()

    def populate_params(self):
        # helper to fetch required parameters
        def _req(key):
            if key not in self.plate_params:
                raise KeyError(f"Missing required plate parameter '{key}'")
            return self.plate_params[key]

        self.Lx = float(_req('Lx'))
        self.Ly = float(_req('Ly'))
        self.h  = float(_req('h'))
        self.T0 = float(_req('T0'))
        self.rho = float(_req('rho'))
        self.E  = float(_req('E'))
        self.nu = float(_req('nu')) 
        self.T60_DC = float(_req('T60_DC'))
        self.T60_F1 = float(_req('T60_F1'))
        self.loss_F1 = float(_req('loss_F1'))
        self.fp = []
        self.op = []
        self.fp.append(float(_req('fp_x')))
        self.fp.append(float(_req('fp_y')))
        self.op.append(float(_req('op_x')))
        self.op.append(float(_req('op_y')))


    def setup(self):
        """
        Setup internal coefficient arrays for modal scheme.
        NOTE: Setup _must_ be run before running the update method if any of the parameters have been changed
        """
        # ---------------------------------------------------------
        
        self.D = self.E * self.h**3 / (12 * (1 - self.nu**2))
        ms = 0.25 * self.rho * self.h * self.Lx * self.Ly  # Modal mass
        self.maxOm = self.fmax * 2 * np.pi

        # Rayleigh damping coefficients
        OmDamp1 = 0.0
        OmDamp2 = 2 * np.pi * self.loss_F1
        dOmSq = OmDamp2**2 - OmDamp1**2
        alpha = 3 * np.log(10) / dOmSq * (OmDamp2**2 / self.T60_DC - OmDamp1**2 / self.T60_F1)
        beta = 3 * np.log(10) / dOmSq * (1 / self.T60_F1 - 1 / self.T60_DC)

        self.ov = self.modal_params_calc()
        self.G1vec, self.G2vec, self.Pvec, self.sigma_vec, self.f0_vec = self.modal_arrays_calc(alpha, beta, ms)


    def modal_params_calc(self):

        DDx = int(np.floor(self.Lx / np.pi * np.sqrt((-self.T0 + np.sqrt(self.T0**2 + 4 * self.maxOm**2 * self.rho * self.h * self.D)) / (2 * self.D))))
        DDy = int(np.floor(self.Ly / np.pi * np.sqrt((-self.T0 + np.sqrt(self.T0**2 + 4 * self.maxOm**2 * self.rho * self.h * self.D)) / (2 * self.D))))

        ov = np.zeros((DDx * DDy, 3))
        ind = 0
        for m in range(1, DDx + 1):
            for n in range(1, DDy + 1):
                g1 = (m * np.pi / self.Lx)**2 + (n * np.pi / self.Ly)**2
                g2 = g1 * g1
                gf = self.T0 / (self.rho * self.h) * g1 + self.D / (self.rho * self.h) * g2
                gf = np.sqrt(gf)
                ov[ind, :] = [gf, m, n]
                ind += 1

        ov[:, 0] = np.where(ov[:, 0] < 20 * 2 * np.pi, self.maxOm + 1000, ov[:, 0])
        ov = ov[np.argsort(ov[:, 0])]
        ov = ov[ov[:, 0] <= self.maxOm]
        return ov


    def modal_arrays_calc(self, alpha, beta, ms):

        DIM = self.ov.shape[0]
        k = 1.0 / self.sample_rate
        G1vec, G2vec, Pvec = np.zeros(DIM), np.zeros(DIM), np.zeros(DIM)
        sigma_vec = np.zeros(DIM)
        f0_vec = np.zeros(DIM)

        for m in range(DIM):
            omref, mind, nind = self.ov[m]
            InWeight = np.cos(self.fp[0] * np.pi * mind) * np.cos(self.fp[1] * np.pi * nind)
            OutWeight = np.cos(self.op[0] * np.pi * mind) * np.cos(self.op[1] * np.pi * nind)
            sig = alpha + beta * omref**2
            G1vec[m] = 2 * np.cos(omref * k) * np.exp(-sig * k)
            G2vec[m] = np.exp(-2 * sig * k)
            Pvec[m] = OutWeight * InWeight * k**2 * np.exp(-sig * k) / ms
            sigma_vec[m] = sig
            f0_vec[m] = omref / (2 * np.pi)
        return G1vec, G2vec, Pvec, sigma_vec, f0_vec


    def IR_time_int(self, Ts, velCalc=False):
        '''
        Modal time integration for impulse response calculation
        :param Ts: Number of time steps
        :param velCalc: If true, calculate velocity output
        :return: Output impulse response (in displacement or velocity)
        '''

        DIM = len(self.G1vec)
        k = 1.0 / self.sample_rate
        q1, q2 = np.zeros(DIM), np.zeros(DIM)
        y = np.zeros(Ts)
        yPrev = 0
        for n in range(Ts):
            fin = 1.0 if n == 0 else 0.0
            q = self.G1vec * q1 - self.G2vec * q2 + self.Pvec * fin
            yCur = np.sum(q1)
            if velCalc:
                y[n] =  (yCur - yPrev) / k
            else:
                y[n] = yCur
            q2, q1, yPrev = q1, q, yCur
        return y

    def synthesize_ir_method(self, duration=1.0, velCalc=False, normalize=True):
        """
        Synthesize plate audio using the IR_time_int method (displacement-based).
        This is the method used in main.py.
        
        Args:
            duration: Duration in seconds
            normalize: Whether to normalize the output
            
        Returns:
            np.array: Synthesized audio signal
        """
        num_samples = int(self.sample_rate * duration)
        out = self.IR_time_int(num_samples, velCalc=velCalc)
        
        if normalize:
            out = out / (1e-8 + np.max(np.abs(out)))  # Normalize
        return out

    @classmethod
    def synthesize_from_params(cls, param_dict, duration=1.0, method='ir', sample_rate=44100, normalize=True):
        """
        Class method to synthesize plate audio directly from parameter dictionary.
        This provides a unified interface for any script requiring quick synthesis of IR from parameters.
        
        Args:
            param_dict: Dictionary of plate parameters
            duration: Duration in seconds
            method: 'ir' for displacement or 'velocity' for velocity output
            sample_rate: Sample rate in Hz
            normalize: Whether to normalize the output
            
        Returns:
            np.array: Synthesized audio signal
        """
        plate = cls(sample_rate=sample_rate, plate_params=param_dict)
        
        if method == 'ir':
            return plate.synthesize_ir_method(duration=duration, normalize=normalize)
        elif method == 'velocity':
            return plate.synthesize_ir_method(duration=duration, velCalc=True, normalize=normalize)
        else:
            raise ValueError(f"Unknown synthesis method: {method}. Use 'ir' or 'velocity'.")

# ---------------------------------------------------------


if __name__ == '__main__':
    plate = ModalPlate() # Init with default parameters
    num_samples = 44100* 5

    start_time = time.time()
    out = plate.synthesize_ir_method()
    elapsed_time = time.time() - start_time
    print(f"Computed {num_samples} samples in {elapsed_time:.2f} seconds.")

    print("Max Amplitude: ", np.max(np.abs(out)))
    os.makedirs('audio_output', exist_ok=True)
    try:
        sf.write('audio_output/plate-ir.wav', out, int(plate.sample_rate))
    except ImportError:
        write('audio_output/plate-ir.wav', int(plate.sample_rate), out.astype(np.float32))
    print("Wrote audio_output/plate-ir.wav")
