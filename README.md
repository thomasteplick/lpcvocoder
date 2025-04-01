<h3>Synthesized Speech using Linear Prediction Coding (LPC) and the Levinson-Durbin Recursion</h3>
<p>
This is a web application written in Go that make use of the html/template package to dynamically create the web page.
Start the web server at bin\speechLPC.exe and connect to it from your web browser at http://127.0.0.1:8080/speechLPCtest.
The program reads in audio WAV files and creates synthesized speech using Linear Prediction Coding (LPC) and the 
Levinson-Durbin Recursion to calculate the predictor coefficients.
</p>
<p>
The checkbox <i>New Speech</i> allows you to create a new wav file by speaking into
the machine's microphone.  The checkbox <i>Words Only</i> will silence audio that 
doesn't contain any speech.  For spectrogram plots, this will eliminate the noise, while
for LPC, it will zero output for those frame samples that don't contain speech.  The checkbox
<i>LPC</i> runs LPC vocoder processing and creates a synthesized speech wav file that will be 
displayed in the spectrogram or the time domain.  The synthesized speech will also be heard 
through your machines audio system.  If LPC is not selected, the original speech wav file will
be displayed and heard.
</p>

<p>
The radio button <i>Time Response</i> displayed the time domain of either the speech or 
synthesized speech wav file.  the <i>Spectrogram</i> radio button displays frequency 
versus time of the speech or synthesized speech.
</p>

<p>
For the Frequency Domain Parameters, select the window type for the Discrete Fourier 
Transform and the size of the transform.  The FFT size is fixed at 256 and the window
type should be Rectangle.
</p>

<p>
For the Speech Parameters, the <i>Threshold</i> determines the audio amplitude at which
speech is detected.  The <i>Window</i> is the integration time over which audio energy
is summed to determine if a word is present.  If enough energy is present in the window, 
then a word is present.
</p>

<p>
The LPC Vocoder Parameters has the <i>Predictor Order</i> which is the number of coefficients
used in the FIR predictor.  The <i>Frame Size</i> is the number of samples used to analyze 
the speech and synthesize the audio.  The time in milliseconds is multiplied by the sample rate
to give the number of samples to use.
</p>

<p>
After the above selections are made, click Submit button and the program will display the selected
waveform and you will hear the audio through your machine's audio system.
</p>

<h3>Waveform Processing</h3

<b>Analysis</b>
<p>
The principle parts of Linear Prediction Coding are an analysis phase and a synthesize phase.  During
the analysis phase, the autocorrelation of the input speech is calculated for each sample frame.  The 
speech is broken up into frames of 80 to 320 samples during which the two phases are run.  After the 
autocorrelation is run, the predictor coefficients and the gain are calculated.  The Levinson-Durbin
recursive algorithm is used to efficiently do the calculations.  The filter order determines the number
of coefficients.  The last calculation done during analysis is voiced or unvoiced determination and
the pitch if voiced is determined. These will give the type of excitation supplied to the model during 
the synthesis phase.  If voiced is used, an impulse train at the pitch period is input the LPC model.
If unvoiced is used, white noise is the input.
</p>

<b>Synthesis</b>
<p>
The voiced or unvoiced excitation is input to the LPC model consisting of the predictor coefficients and
te gain.  Again this done for each input frame.  That is a different model is used for each frame along
with the corresponding excitation.
</p>


                                      
