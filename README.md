<h3>Configuration for Time, Frequency, and LPC Parameters</h3>
<p>
The checkbox <i>New speech</i> allows you to create a new wav file by speaking into
the machine's microphone.  The checkbox <i>Words Only</i> will silence audio that 
doesn't contain any speech.  For spectrogram plots, this will eliminate the noise, while
for LPC, it will zero output for those frame samples that don't contain speech.  The checkbox
<i>LPC</i> runs LPC vocoder processing and creates a synthesized speech wav file that will be 
displayed in the spectrogram or the time domain.  The synthesized speech will also be heard 
through your machines audio system.  If LPC is not selected, the original speech wav file will
be displayed and heard.  To hear the speech on your audio device, the <b>fmedia</b> program should
be installed on your machine and the path inserted in your PATH environment variable.
</p>
<p>
The radio button <i>Time Response</i> displays the time domain of either the speech or 
synthesized speech wav file.  Yhe <i>Spectrogram</i> radio button displays frequency 
versus time of the speech or synthesized speech.  Shades of gray signify the power
present at that frequency; the darker the color, the greater the power at that frequency.
</p>
<p>
For the Frequency Domain Parameters, select the window type for the Discrete Fourier 
Transform and the size of the transform.  The FFT size is fixed at 256 and the window
type should be Rectangle.
</p>
<p>
For the Speech Parameters, the <i>Threshold</i> determines the audio amplitude at which
speech is detected.  The <i>Window</i> is the integration time over which audio power
is summed (energy) to determine if a word is present.  If enough energy is present in the window, 
then a word is present.
</p>
<p>
The LPC Vocoder Parameters has the <i>Predictor Order</i>, which is the number of coefficients
used in the FIR predictor.  The <i>Frame Size</i> is the number of samples used to analyze 
the speech and synthesize the audio.  The time in milliseconds is multiplied by the sample rate
to give the number of samples to use.
</p>
<p>
After the above selections are made, click Submit button and the program will display the selected
waveform and you will hear the audio through your machine's audio system.
</p>
<h3>Speech Signal Processing</h3
                              
<b>Analysis</b>
<p>
The principle parts of Linear Prediction Coding are an analysis phase and a synthesis phase.  During
the analysis phase, the autocorrelation of the input speech is calculated for each sample frame.  The 
speech is broken up into frames of 80 to 320 samples during which the two phases are run.  After the 
autocorrelation is run, the predictor coefficients and the gain are calculated.  The Levinson-Durbin
recursive algorithm is used to efficiently do the calculations.  This recursive algorithm is an efficient
way to solve the normal equations, which are the Yule-Walker equations.  The filter order determines the number
of coefficients.  The last calculation done during analysis is voiced or unvoiced determination and
the pitch if voiced is determined. These will give the type of excitation supplied to the model during 
the synthesis phase.  If voiced excitation is used, an impulse train at the pitch period is input to the LPC model.
If unvoiced excitation is used, white noise is the input.
</p>

<b>Synthesis</b>
<p>
The voiced or unvoiced excitation is input to the LPC model consisting of the predictor coefficients and
the gain.  Again this is done for each input frame.  That is, a different model is used for each frame along
with the corresponding excitation. This is so because the speech signal is time-varying and non-stationary
whose spectral characteristics change in 10-50 msec time intervals.  Therefore each time interval requires
separate analysis and synthesis.  The synthesis filter is essentially the inverse of the 
FIR predictor.  It is an all-pole filter with all poles inside the unit circle, hence it is stable.  The filter
order determines the number of poles.
</p>
