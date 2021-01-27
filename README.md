# Filtering-signal
Adaptive Noise Cancellation refers to the application of Adaptive filtering
algorithms to noise cancellation. Noise cancellation in signal processing means
removal of unwanted noise or unwanted frequencies from the signal. Active
Noise Cancellation makes use of Adaptive Filtering, which is a filtering method
that looks at sampling audio waves analysing the input audio for noise and then
produces a destructive wave of the same frequency and amplitude as the noise
present in the audio wave which results in cancellation of the noise. We
implement three algorithms.
- Least Mean Squares (LMS), Normalized Least
- Mean Squares (NLMS) and 
- Recursive Least Squares (RLS). 

We compare the performance of the three adaptive filters.
We designed a Graphical User Interface (GUI) using Matlab where we plot various graphs for showing the
transition of the input audio from an unfiltered noisy signal to a filtered signal.


### 1. Least Mean Squares (LMS) Adaptive Filter:

LMS Algorithm is derived from the steepest descent algorithm. We find out the gradient at every iteration. LMS filters find the filter weights by producing the least mean square squares of the error signal. In this algorithm, the filter is updated based on the error at the given current time.The LMS Algorithm iteratively changes the weights of a Finite Response Filter (FIR).
The output signal, y(k) is calculated as follows :

Where,
 - u(k) : input filter vector
 - w(k) : weights (a FIR filter)
 - 𝜇 : fixed step size of the filter
 - d(k) : reference signal to the adaptive filter

Error signal [e(k)] can be calculated as :

*e(k) = d(k) - y(k)*

Filter coefficients are updated based on e(k) :

Larger values of 𝜇 will increase the adaptation rate but it will also lead to
increase in residual mean-squared error.


### 2. Normalized Least Mean Squares (NLMS)

NLMS is similar to the LMS Algorithm. NLMS varies from LMS due to the
difference in step size. LMS is not scalable with increasing input. It becomes
really difficult to choose a learning rate μ that guarantees stability of the
algorithm. In NLMS, the step size is adapted according to the signal power and
amplitude.
The output vector is calculated as :
Where,
 - w(n) : current weight
 - v2(n) : input signal
 - y(n) : output signal
Error signal is calculated as :
 - e(n) = d(n) - y(n)
Where,
e(n) : error signal
d(n) : desired signal
The weight in NLMS algorithm is :
We consider both Mean Square Error and Signal-to-Noise Ratio for better
comparison.

### 3. Recursive Least Squares (RLS)

RLS is an algorithm that finds the filter weights recursively so as to minimize a
weighted linear least squares cost function with respect to the input signals. RLS
works best in time varying environments but has very high complexity at the
same time. In RLS, we need to know the estimate of previous samples of output
signal, error signal and filter weight and hence, higher memory requirements.
The weight vector is updated using the following equations. Where ‘k’ is the
Kalman Gain . RLS works similar to a Kalman Filter. It gives an estimate of the
state of the system as an average of the system’s predicted state and of the new
measurement using a weighted average. Kalman Gain is the relative weight
given to the measurements and the current
