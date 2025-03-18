/*
Package lpcvocoder is a Linear Predictive Coding Vocoder for speech analysis and synthesis
using the Levinson-Durbin recursion to solve the Yule-Walker equations.
*/
package lpcvocoder

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path"
	"path/filepath"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
)

const (
	dataDir    = "data/"           // directory for the weights and audio wav files
	maxSamples = 10000             // max audio wav samples = 1.158 sec * sampleRate
	sampleRate = 8000              // Hz or samples/sec
	bitDepth   = 16                // audio wav encoder/decoder sample size
	msgPredWav = "messagePred.wav" // Synthesized speech of message/subject wav file
)

type Vocoder struct {
	frameSize int          // number of samples
	predOrder int          // number of coefficients in predictor
	a         [2][]float64 // previous order and current order predictor coefficients
	G         float64      // Gain
	R         []float64    // autocorrelation of speech
	// silence bool         // mute the predicted speech
	file       string
	pitch      int       // frequency of the frame for voiced
	predSpeech []float64 // modeled output
	speech     []float64 // input
}

// determinePitch decides if the frame speech is voiced or unvoiced and the pitch if voiced
func (vcdr *Vocoder) determinePitch() error {
	sum := 0.0
	cur := 0
	count := 0
	vcdr.pitch = 0
	npk := 0
	pk := 0.0
	const delta = 2.0
	// Find first peak
	for i := cur + 1; i < vcdr.frameSize/2; i++ {
		if vcdr.R[i] > pk {
			npk = i
			pk = vcdr.R[i]
		}
	}
	sum += float64(npk - cur)
	count++
	vcdr.pitch = int(sum / float64(count))
	cur = npk
	// Find any remaining peaks if enough samples are left
done:
	for (vcdr.frameSize/2 - cur) > vcdr.pitch {
		pk = 0.0
		for i := cur + 1; i < vcdr.frameSize/2; i++ {
			if vcdr.R[i] > pk {
				npk = i
				pk = vcdr.R[i]
			}
		}
		// Determine if the same period is found
		if pk > 0 && (math.Abs(float64(npk-cur)) <= delta) {
			sum += float64(npk - cur)
			count++
			vcdr.pitch = int(sum / float64(count))
			cur = npk
			// this peak has a different period than the average
		} else {
			if count < 2 {
				vcdr.pitch = 0
			}
			break done
		}
	}
	return nil
}

// genModel generates the FIR predictor coefficients and the Gain
func (vcdr *Vocoder) genModel() error {
	E := vcdr.R[0]
	p := vcdr.predOrder
	cur := 1
	prev := 0
	for i := 1; i <= p; i++ {
		k := vcdr.R[i]
		for j := 1; j < i; j++ {
			k -= vcdr.a[prev][j] * vcdr.R[i-j]
		}
		k /= E
		vcdr.a[cur][i] = k
		for j := 1; j < i; j++ {
			vcdr.a[cur][j] = vcdr.a[prev][j] - k*vcdr.a[prev][i-j]
		}
		E = (1.0 - k*k) * E
		cur, prev = prev, cur
	}
	vcdr.G = math.Sqrt(E)
	return nil
}

// gemAutoCorr generates the autocorrelation of the input speech
func (vcdr *Vocoder) genAutoCorr(frame int) error {
	n := frame * vcdr.frameSize
	for k := 0; k < vcdr.frameSize/2; k++ {
		sum := 0.0
		for m := 0; m < vcdr.frameSize-k; m++ {
			sum += vcdr.speech[n+m] * vcdr.speech[n+m+k]
		}
		vcdr.R[k] = sum
	}
	return nil
}

// creates a model for the speech consisting of predictor coefficients and gain G
func (vcdr *Vocoder) analyze(frame int) error {

	/*
		Check for silence by finding word boundaries.  If the frame beginning or frame
		ending is not within any word boudary, then set silence to true.  Skip the rest
		of analysis.  In synchesize, set predSignal to zero for the frame.
	*/

	// Compute autocorrelation R for frame size / 2 lags
	err := vcdr.genAutoCorr(frame)
	if err != nil {
		fmt.Printf("vcdr.genAutoCorr error: %v\n", err.Error())
		return fmt.Errorf("vcdr.genAutoCorr error: %v", err.Error())
	}

	// Compute predictor coefficients and Gain using Levinson-Durbin recursion to
	// solve the Yule-Walker equations
	err = vcdr.genModel()
	if err != nil {
		fmt.Printf("vcdr.genModel error: %v\n", err.Error())
		return fmt.Errorf("vcdr.genModel error: %v", err.Error())
	}

	// Determine if voiced or unvoiced excitation using autocorrelation R
	// Determine pitch if voiced using the autocorrelation R
	err = vcdr.determinePitch()
	if err != nil {
		fmt.Printf("vcdr.determinePitch error: %v\n", err.Error())
		return fmt.Errorf("vcdr.determinePitch error: %v", err.Error())
	}

	return nil
}

// synthesize creates estimated speech using the predictor coefficients and Gain
// found in analysis
func (vcdr *Vocoder) synthesize(frame int) error {
	// Using the predictor coefficents a[k] and gain G found in analyze, synthesize speech
	// for the given frame samples.  y(n) = sum{a(k)*y(n-k)} + G*u(n)
	// The excitation u(n) is either an impulse with pitch period or white noise (voiced or unvoiced)

	// Check for silence and set vcdr.predSpeech[] to zero for this frame

	n := frame * vcdr.frameSize
	u := 0.0
	sum := 0.0
	// Determine which coefficient slice has the final order p since
	// it is toggling back and force in genModel
	k := vcdr.predOrder % 2
	if n == 0 {
		for i := n; i < vcdr.predOrder; i++ {
			if vcdr.pitch > 0 {
				if (i % vcdr.pitch) == 0 {
					u = vcdr.G * 1.0
				} else {
					u = 0.0
				}
			} else {
				u = vcdr.G * (rand.Float64() - 0.5)
			}
			sum = u
			for j := 1; j <= i; j++ {
				sum += vcdr.a[k][j] * vcdr.predSpeech[i-j]
			}
			vcdr.predSpeech[i] = sum
		}
		n += vcdr.predOrder
	}
	// Continue from last sample above
	for i := n; i < n+vcdr.frameSize; i++ {
		// voiced speech uses an impulse with pitch period
		if vcdr.pitch > 0 {
			if (i % vcdr.pitch) == 0 {
				u = vcdr.G * 1.0
			} else {
				u = 0.0
			}
		} else {
			u = vcdr.G * (rand.Float64() - 0.5)
		}
		sum = u
		for j := 1; j <= vcdr.predOrder; j++ {
			sum += vcdr.a[k][j] * vcdr.predSpeech[i-j]
		}
		vcdr.predSpeech[i] = sum
	}
	return nil
}

// Perform analysis and synthesis of the input speech signal in frameSize blocks
func ProcessSpeech(fileName string, frameSize int, predOrder int) error {
	// new Vocoder object
	vcdr := Vocoder{file: fileName, frameSize: frameSize, predOrder: predOrder}
	vcdr.R = make([]float64, frameSize/2)
	// leading coefficient a[0][0] and a[1][0] = 1.0 and not used
	vcdr.a[0] = make([]float64, predOrder+1)
	vcdr.a[1] = make([]float64, predOrder+1)

	// open speech WAV file and convert 16-bit samples to []float64
	// Open the testing message
	f, err := os.Open(filepath.Join(dataDir, fileName))
	if err != nil {
		fmt.Printf("Open file %s error: %v", fileName, err)
		return fmt.Errorf("open file %s error: %s", fileName, err.Error())
	}
	defer f.Close()

	// Create wav Decoder, intBuf, fltBuf and Decode the wav file
	dec := wav.NewDecoder(f)
	bufInt := audio.IntBuffer{
		Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate},
		Data:   make([]int, 2*maxSamples), SourceBitDepth: bitDepth}
	nsamples, err := dec.PCMBuffer(&bufInt)
	if err != nil {
		fmt.Printf("PCMBuffer error: %v\n", err)
		return fmt.Errorf("PCMBuffer error: %v", err.Error())
	}
	vcdr.speech = bufInt.AsFloatBuffer().Data
	vcdr.predSpeech = make([]float64, nsamples)
	//fmt.Printf("%s samples = %d\n", filename, nsamples)

	// loop over frames: #samples/frameSize
	nframes := nsamples / vcdr.frameSize
	for frame := 0; frame < nframes; frame++ {
		err := vcdr.analyze(frame)
		if err != nil {
			fmt.Printf("ProcessSpeech analyze error: %v\n", err)
			return fmt.Errorf("ProcessSpeech analyze error %v", err.Error())
		}
		err = vcdr.synthesize(frame)
		if err != nil {
			fmt.Printf("ProcessSpeech synthesize error: %v\n", err)
			return fmt.Errorf("ProcessSpeech synthesize error %v", err.Error())
		}
	}

	// Create new wav file: save synthesized speech msgPredWav to disk
	outF, err := os.Create(path.Join(dataDir, msgPredWav))
	if err != nil {
		fmt.Printf("os.Create() file %s error: %v\n", msgPredWav, err)
		return fmt.Errorf("os.Create() file %s error: %v", msgPredWav, err)
	}
	defer outF.Close()

	// create wav.Encoder
	enc := wav.NewEncoder(outF, sampleRate, bitDepth, 1, 1)

	// create audio.FloatBuffer
	float64Buf := &audio.FloatBuffer{Data: vcdr.predSpeech, Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate}}

	// create IntBuffer from FloatBuffer and pass to Encoder.Write()
	if err := enc.Write(float64Buf.AsIntBuffer()); err != nil {
		fmt.Printf("wav encoder write error: %v\n", err)
		return fmt.Errorf("wav encoder write error: %v", err.Error())
	}

	// close the encoder
	if err := enc.Close(); err != nil {
		fmt.Printf("wav encoder close error: %v\n", err)
		return fmt.Errorf("wav encoder error: %v", err.Error())
	}

	return nil
}
