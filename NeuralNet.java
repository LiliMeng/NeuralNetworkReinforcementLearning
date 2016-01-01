package Sarb;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.Random;

/**
 * @author sarbjit
 * {@docRoot}
 * @version 0.14
 * This class implements a feed forward multi-layer perceptron.
 * It provides methods for both constructing & training a neural net.
 * The network is fixed at a single hidden layer and a single output neuron
 * but the number of hidden neurons and the number of inputs is variable.
 * 
 * v0.2 adds support for tracing the largest and smallest Q value requested
 * to train. See minQ and maxQ.
 * v0.3 corrects some spelling mistakes
 * v0.4 adds load and save methods
 * v0.5 minor fix - argA & arbB should be doubles.
 * v0.6 Bug in v0.5. Was not saving/loading the bias weights!
 * v0.7 Add close to save method
 * v0.8 Made the numHidden, numInputs, rho and alpha public attributes
 * v0.9 Upto version 0.8 during training a delta was being computed for the output and 
 *      used to update the weights to the output neuron *before* computing and adjusting
 *      the weights in the layer below. We fix this now so the weights of the hidden neuron
 *      are updated after the updates have been made to the hidden layer. i.e. It is more correct
 *      to compute all deltas before updating the weights.
 * v0.10 Reverted back change in v0.9. I found that this introduce more occurrences where
 * 	    the network does not converge when testing agains the XOR.
 * v0.11 Added check to see if during training, target requested is outside bounds of argA-argB
 * 		 PLus fixed a typo. "arbB" was used instead of "argB"
 * V0.12 Code changed to use an interface
 * v0.13 If number of hidden units or inputs do not match when loading file, code now throws exception rather
 * 		 than silently failing. (Error will appear in Robocode console)
 * v0.14 25 Oct 2012: Update to enable the hidden neurons to b switched from using
 * 		 a binary sigmoid or biploar sigmoid
 * v0.15 28 Oct 2012. There was a bug in the computation of the output of the hidden neurons.
 * 		 The bias computation of the hidden neurons in the outputFor() method was incorrectly
 * 		 placed in the inner loop. It should have been in the outer loop. Thanks for Ori Hadary for
 * 		 spotting that!
 */
public class NeuralNet implements NeuralNetInterface {
	
	/**
	 * Private members of this class
	 */
	private double [] hiddenNeuron; 					// hidden neuron vector	
	private double [] weightHiddenToOutput;				// weight vector from hidden layer to output; [0] is  
	private double [][] weightInputToHidden;			// weight vector from input layer to hidden; [0][0] is bias
	
	private double [] lastWeightChangeHiddenToOutput;	// Used to implement the momentum term alpha
	private double [][] lastWeightChangeInputToHidden;	// Used to implement the momentum term alpha
	
	private double a,b;									// Sigmoid asmyptotes
	private double b_minus_a;							// These terms from Fausett pg 309
	private double minus_a;								// Used to generate output in the range (a,b)
	
	final double bias = 1.0;							// The input for each neurons bias weight
	
	private boolean useBipolarAtHiddenLayer = false;	// To toggle if hidden neurons use biploar. Otherwise use binary sigmoid.
	

	/**
	 * Public attributes of this class. Used to capture the largest Q
	 * value requested to train and the smallest.
	 * They are static so are preserved over multiple rounds.
	 */
	static public double maxQ = 0.0;
	static public double minQ = 0.0;
	public int numInputs;								// number of possible input
	public int numHidden;								// number of hidden neurons
	public double rho;									// learning rate for NN
	public double alpha;								// alpha term for NN
	
	/**
	 * Constructor.
	 * @param argNumInputs The number of inputs in your input vector
	 * @param argNumHidden The number of hidden neurons in your hidden layer. Only a single hidden layer is supported
	 * @param argLearningRate The learning rate coefficient
	 * @param argMomentumTerm The momentum coefficient
	 * @param argA Integer lower bound of sigmoid used by the output neuron only.
	 * @param argB Integer upper bound of sigmoid used by the output neuron only.
	 * @param argBipolarHidden Selects whether to use a bipolar or binary sigmoid for the hidden layer
	 * @param argUseBipolarHIddenNeurons Flag to indicate if hidden neurons should be bipolar or boolean
	 */
	public NeuralNet (
			int argNumInputs,
			int argNumHidden,
			double argLearningRate,
			double argMomentumTerm,
			double argA,
			double argB,
			boolean argUseBipolarHIddenNeurons) 
	{
		System.out.println( "--* NN constructor called..." );
		
		numInputs = argNumInputs;
		numHidden = argNumHidden;
		rho = argLearningRate;
		alpha = argMomentumTerm;
		minQ = argA;
		maxQ = argB;
		hiddenNeuron = new double [numHidden];
		weightHiddenToOutput = new double [numHidden + 1]; 				// +1 for bias input
		weightInputToHidden = new double [numHidden][numInputs + 1];	// +1 for bias input
		useBipolarAtHiddenLayer = argUseBipolarHIddenNeurons;
		
		// These are used to implement the momentum term
		lastWeightChangeHiddenToOutput = new double [numHidden + 1];	
		lastWeightChangeInputToHidden = new double [numHidden][numInputs + 1];
		
		// These are used to scale the range of the sigmoid to any general range (a,b)
		// See Fausett pg 309 for details
		a = argA;
		b = argB;
		b_minus_a = b-a;
		minus_a = -a;

		// Randomly initialize weights.
		initializeWeights ();
		// zeroWeights ();
	}
	
	/**
	 * Return sigmoid of the input X
	 * @param x The input
	 * @return either a binary or a bipolar sigmoid depending upon flag.
	 */
	public double sigmoid( double x ) {
		if (useBipolarAtHiddenLayer)
			return bipolarSigmoid(x);
		else
			return binarySigmoid(x);
	}

	/**
	 * Return a bipolar sigmoid of the input X
	 * @param x The input
	 * @return f(x) = 2 / (1+e(-x)) - 1
	 */
	public double bipolarSigmoid( double x ) {
		return 2 / (1 + Math.exp(-1 * x)) - 1;
	}
	
	/**
	 * Return a binary sigmoid of the input X
	 * @param x The input
	 * @return f(x) =  1 / (1+e(-x))
	 */
	public double binarySigmoid( double x ) {
		return 1 / (1 + Math.exp(-1 * x));
	}
	
	/**
	 * This method implements a general sigmoid with asymptotes bounded by (a,b)
	 * @param x The input
	 * @return f(x) = b_minus_a / (1 + e(-x)) - minus_a
	 */
	public double customSigmoid( double x ) {
		return b_minus_a / (1 + Math.exp(-1 * x)) - minus_a;
	}
	
	/**
	 * Initialize the weights to random values.
	 * For say 2 inputs, the input vector is [0] & [1]. We add [2] for the bias.
	 * Like wise for hidden units. For say 2 hidden units which are stored in an array.
	 * [0] & [1] are the hidden & [2] the bias.
	 * We also initialise the last weight change arrays. This is to implement the alpha term.
	 */
	public void initializeWeights() {	
		Random rand = new Random ();
			for ( int i=0; i<numHidden; i++ ) {
				weightHiddenToOutput [i] = rand.nextDouble () - 0.5;
				lastWeightChangeHiddenToOutput [i] = 0;
			}
			weightHiddenToOutput [numHidden] = rand.nextDouble () - 0.5; 		// Init bias weight to output neuron.
			lastWeightChangeHiddenToOutput [numHidden] = 0;
				
			for ( int i=0; i<numHidden; i++ ) {
				for ( int j=0; j<numInputs; j++) {
					weightInputToHidden [i][j] = rand.nextDouble () - 0.5;
					lastWeightChangeInputToHidden [i][j] = 0;
				}
				weightInputToHidden [i][numInputs] = rand.nextDouble () - 0.5;	// Init bias weight to each hidden neuron.
				lastWeightChangeInputToHidden [i][numInputs]= 0;
			}
		}

	/**
	 * Initialize the weights to 0.
	 */
	public void zeroWeights() {	
			for ( int i=0; i<numHidden; i++ ) {
				weightHiddenToOutput [i] = 0;
				lastWeightChangeHiddenToOutput [i] = 0;
			}
			weightHiddenToOutput [numHidden] = 0; 		// Init bias weight to output neuron.
			lastWeightChangeHiddenToOutput [numHidden] = 0;
				
			for ( int i=0; i<numHidden; i++ ) {
				for ( int j=0; j<numInputs; j++) {
					weightInputToHidden [i][j] = 0;
					lastWeightChangeInputToHidden [i][j] = 0;
				}
				weightInputToHidden [i][numInputs] = 0;	// Init bias weight to each hidden neuron.
				lastWeightChangeInputToHidden [i][numInputs]= 0;
			}
		}

	
	/**
	 * Computes output of the NN without training. I.e. a forward pass
	 * @param X The input vector. An array of doubles.
	 * @return Sigmoid thresholded output value. 0 if length of input vector is wrong.
	 */
	public double outputFor ( double[] X ) {	
		
		double [] weightedSumHidden = new double [numHidden];
		double weightedSumOutput;
		
		if ( X.length != numInputs ) {
			System.out.println( "-** Length of input vector expected: " + numInputs + "Got: " + X.length );
			return 0;
		}
		
		// Compute weighted sum at hidden neurons
		for ( int i=0; i<numHidden; i++ ) {
			weightedSumHidden [i] = 0;
			for ( int j=0; j<numInputs; j++ ) {
				weightedSumHidden [i] += X[j] * weightInputToHidden [i][j];
			}
			weightedSumHidden [i] += bias * weightInputToHidden [i][numInputs];
			hiddenNeuron [i] = sigmoid ( weightedSumHidden [i] );
		}
		
		// Compute weighted sum of output neuron
		weightedSumOutput = 0;
		for ( int i=0; i<numHidden; i++ )
			weightedSumOutput += hiddenNeuron [i] * weightHiddenToOutput [i];
		weightedSumOutput += bias * weightHiddenToOutput [numHidden];
	
		// We have the final output. Return it.
		return customSigmoid ( weightedSumOutput );
	}
	
	/**
	 * This method is used to update the weights of the neural net.
	 * @param argInputVector The input vector
	 * @param argTargetOuput The expected output
	 * @return error The error used to train. (I.e. error before the update)
	 */
	public double train ( double [] argInputVector, double argTargetOutput )
	{
		double actualOutput = outputFor ( argInputVector );
		double errorAtOutput = argTargetOutput - actualOutput;
		
		// Derivative for a general sigmoid bounded by the range (a,b). See Fausett pg 309 for details.
		double sigmoidPrimeAtOutput = (1 / b_minus_a ) * (minus_a + actualOutput) * (b_minus_a - minus_a - actualOutput); 
		double deltaAtOutput = errorAtOutput * sigmoidPrimeAtOutput; // delta is the error signal
		
		double weightChange = 0;
		
		if (argTargetOutput > b) System.out.println ("*** NeuralNet: Target of " + argTargetOutput + " exceeds upper bound of " + b);
		if (argTargetOutput < a) System.out.println ("*** NeuralNet: Target of " + argTargetOutput + " exceeds lower bound of " + a);
		
		// Update minQ and maxQ if necessary
		if ( argTargetOutput > maxQ ) maxQ = argTargetOutput;
		if ( argTargetOutput < minQ ) minQ = argTargetOutput;

		// Now update the weights of the output neuron
		for ( int i=0; i<numHidden; i++) {
			weightChange = alpha * lastWeightChangeHiddenToOutput [i] + rho * deltaAtOutput * hiddenNeuron [i];
			weightHiddenToOutput [i] += weightChange;
			lastWeightChangeHiddenToOutput [i] = weightChange; // Update for next use of momentum
		}
		// Also update the weight on the bias
		weightChange = alpha * lastWeightChangeHiddenToOutput [numHidden] + rho * deltaAtOutput * bias;
		weightHiddenToOutput [numHidden] += weightChange;
		lastWeightChangeHiddenToOutput [numHidden] = weightChange; // Update for next use of momentum		
		
		// Update the hidden layer of neurons. We really want to do this before we update
		// the weights going from the hidden to the output neuron but this seems to work better.
		double deltaAtHidden = 0;
		double sigmoidPrimeAtHidden = 0;
		for ( int i=0; i<numHidden; i++) {
			// First compute  error signals for the hidden neuron
			if (useBipolarAtHiddenLayer)
				sigmoidPrimeAtHidden = 0.5 * (1 + hiddenNeuron [i]) * (1 - hiddenNeuron [i]);	// derivative for a bipolar sigmoid
			else
				sigmoidPrimeAtHidden = hiddenNeuron [i] * (1 - hiddenNeuron [i]);				// derivative for a binary sigmoid
				
			deltaAtHidden = sigmoidPrimeAtHidden * weightHiddenToOutput [i] * deltaAtOutput;
	
			// Now update the weights of the neuron
			for ( int j=0; j<numInputs; j++ ) {
				weightChange = alpha * lastWeightChangeInputToHidden [i][j] + rho * deltaAtHidden * argInputVector[j];
				weightInputToHidden[i][j] += weightChange;
				lastWeightChangeInputToHidden [i][j] = weightChange; // Update for next use of momentum
			}
			// Remember to update weights on the biases too.
			weightChange = alpha * lastWeightChangeInputToHidden [i][numInputs] + rho * deltaAtHidden * bias;
			weightInputToHidden[i][numInputs] += weightChange;
			lastWeightChangeInputToHidden [i][numInputs] = weightChange; // Update for next use of momentum
		}
		
		return errorAtOutput;
	}

	/**
	 * Saves the weights to file. Format of the output is as follows:
	 * 		numInputs
	 * 		numHidden
	 * 		weight input 0 to hidden 0
	 * 		weight input 0 to hidden 1
	 * 		.....
	 * 		weight input j to hidden i
	 * 		weight hidden 0 to output
	 * 		weight hidden i to output
	 * @param argFile A "File" handle to where the weights are to be written
	 */
	public void save ( File argFile ) {
		PrintStream saveFile = null;
		
		try {
			saveFile = new PrintStream( new FileOutputStream( argFile ));
		}
		catch (IOException e) {
			System.out.println( "*** Could not create output stream for NN save file.");
		}
		
		saveFile.println( numInputs );
		saveFile.println( numHidden );
		
		// First save the weights from the input to hidden neurons (one line per weight)
		for ( int i=0; i<numHidden; i++) {
			for ( int j=0; j<numInputs; j++) {
				saveFile.println( weightInputToHidden [i][j] );
			}
			saveFile.println( weightInputToHidden [i][numInputs] ); // Save bias weight for this hidden neuron too
		}
		// Now save the weights from the hidden to the output neuron
		for (int i=0; i<numHidden; i++) {
			saveFile.println( weightHiddenToOutput [i] );
		}	
		saveFile.println( weightHiddenToOutput [numHidden] ); // Save bias weight for output neuron too.
		saveFile.close();
	}
	
	/**
	 * Loads the weights from file. Format of the file is expected to follow
	 * that specified in the "save" method specified elsewhere in this class.
	 * @param argFileName the name of the file where the weights are to be found
	 */
	public void load ( String argFileName ) throws IOException {

		FileInputStream inputFile = new FileInputStream( argFileName );
		BufferedReader inputReader = new BufferedReader(new InputStreamReader( inputFile ));		
				
		// Check that NN defined for file matches that created
		int numInputInFile = Integer.valueOf( inputReader.readLine() );
		int numHiddenInFile = Integer.valueOf( inputReader.readLine() );
		// System.out.println("--- File: #inputs=" + numInputInFile + ", #hidden=" + numHiddenInFile);
		// System.out.println("--- NNet: #inputs=" + numInputs + ", #hidden=" + numHidden);
		
		if ( numInputInFile != numInputs ) {
			System.out.println ( "*** Number of inputs in file is " + numInputInFile + " Expected " + numInputs );
			throw new IOException();
		}
		if ( numHiddenInFile != numHidden ) {
			System.out.println ( "*** Number of hidden in file is " + numHiddenInFile + " Expected " + numHidden );
			throw new IOException();
		}
		if ( (numInputInFile != numInputs) || (numHiddenInFile != numHidden) ) return;
		
		// First load the weights from the input to hidden neurons (one line per weight)
		for ( int i=0; i<numHidden; i++) {
			for ( int j=0; j<numInputs; j++) {
				weightInputToHidden [i][j] = Double.valueOf( inputReader.readLine() );
			}
			weightInputToHidden [i][numInputs] = Double.valueOf( inputReader.readLine() ); // Load bias weight for this hidden neuron too
		}
		// Now load the weights from the hidden to the output neuron
		for (int i=0; i<numHidden; i++) {
			weightHiddenToOutput [i] = Double.valueOf( inputReader.readLine() );
		}	
		weightHiddenToOutput [numHidden] = Double.valueOf( inputReader.readLine() ); // Load bias weight for output neuron too.
		
		inputReader.close(); 
	}	

} // End of public class NeuralNet
