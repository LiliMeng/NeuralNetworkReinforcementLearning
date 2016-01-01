package part1_NN;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Vector;

public class NeuralNetwork {
	private int input_num;
	private int output_num;
	private int hidden_num;
	private Vector<InputCell> input_nodes = new Vector<InputCell>(0);
	private ArrayList<OutputCell> output_nodes = new ArrayList<OutputCell>(0);
	private ArrayList<NeuralCell> hidden_nodes = new ArrayList<NeuralCell>(0);
	public static final double THRESHOLD = 0.05;
	public static final double TIMEOUT_LOOP_COUNT = 10000;
	
	private double lower_bound, upper_bound;
	/**
	 * Construct a two-layer NeuralNetwork
	 * @param num_of_input number of inputs
	 * @param num_of_output number of outputs
	 * @param num_of_hidden number of hidden nodes
	 */
	public NeuralNetwork(int num_of_input, int num_of_output, int num_of_hidden)
	{
		this.input_num = num_of_input + 1;
		this.output_num = num_of_output;
		this.hidden_num = num_of_hidden + 1;
		
		for(int i = 0; i<input_num -1; i++)
		{
			input_nodes.add(new InputCell());
		}
		input_nodes.add(new InputCell(1.0));
		
		for(int i = 0; i<hidden_num -1; i++)
		{
			hidden_nodes.add(new NeuralCell(input_nodes));
		}
		hidden_nodes.add(new InputCell(1.0));
		
		for(int i = 0; i<output_num; i++)
		{
			output_nodes.add(new OutputCell(hidden_nodes));
		}
	}

	public double[] outputFor(double[] X) {
		//just computing
		for(int i = 0; i<X.length;i++)
		{
			this.input_nodes.get(i).setInput(X[i]);
		}
		
		
		for(int i = 0; i<this.input_num; i++)
		{
			this.input_nodes.get(i).getSumFromInput();
			this.input_nodes.get(i).getOutput();
		}
		for(int i = 0; i<this.hidden_num; i++)
		{
			this.hidden_nodes.get(i).getSumFromInput();
			this.hidden_nodes.get(i).getOutput();
		}
		double[] output = new double[this.output_num];
		for(int i = 0; i<this.output_num; i++)
		{
			this.output_nodes.get(i).getSumFromInput();
			output[i] = this.output_nodes.get(i).getOutput();
		}
		//so far, we only consider one output situation
		//return output_nodes.get(0).getOutput();
		// this time we need to consider more than one output
		return output;
	}

	public double[] train(double[] X, double[] argValue) {
		//initialise input set
		for(int i = 0; i<X.length;i++)
		{
			this.input_nodes.get(i).setInput(X[i]);
		}
		
		//set correct output
		for(int i = 0; i<this.output_num; i++)
		{
			this.output_nodes.get(i).setCorrectOutput(argValue[i]);
		}
		//computing
		for(int i = 0; i<this.input_num; i++)
		{
			this.input_nodes.get(i).getSumFromInput();
			this.input_nodes.get(i).getOutput();
		}
		for(int i = 0; i<this.hidden_num; i++)
		{
			this.hidden_nodes.get(i).getSumFromInput();
			this.hidden_nodes.get(i).getOutput();
		}
		for(int i = 0; i<this.output_num; i++)
		{
			this.output_nodes.get(i).getSumFromInput();
			this.output_nodes.get(i).getOutput();
		//start to back propagate
			/*this.output_nodes.get(i).updateWeights();
			this.output_nodes.get(i).updateErrorOfLowerCells();
			//this.output_nodes.get(i).updateWeights();
			this.output_nodes.get(i).refresh();*/
		}
		for(int i = 0; i<this.output_num; i++)
		{
			this.output_nodes.get(i).updateWeights();
			this.output_nodes.get(i).updateErrorOfLowerCells();
			//this.output_nodes.get(i).updateWeights();
			this.output_nodes.get(i).refresh();
		}
		for(int i = 0; i<this.hidden_num; i++)
		{
			//this.hidden_nodes.get(i).updateErrorOfLowerCells();
			this.hidden_nodes.get(i).updateWeights();
			this.hidden_nodes.get(i).refresh();
		}
		for(int i = 0; i<this.input_num; i++)
		{
			this.input_nodes.get(i).updateErrorOfLowerCells();
			this.input_nodes.get(i).updateWeights();
			this.input_nodes.get(i).refresh();
		}
		//return output_nodes.get(0).getError();
		double[] error = new double[this.output_num];
		for(int i = 0; i<error.length; i++)
			error[i] = 0;
		for(int i=0; i<this.output_num; i++)
		{
			error[i] = output_nodes.get(i).getError();
		}
		/*System.out.println("output error[0]: "+error[0]+"\tcorrect value: "+argValue[0]);
		System.out.println("output error[1]: "+error[1]+"\tcorrect value: "+argValue[1]);
		System.out.println("output error[2]: "+error[2]+"\tcorrect value: "+argValue[2]);
		System.out.println("output error[3]: "+error[3]+"\tcorrect value: "+argValue[3]);
		System.out.println("output error[4]: "+error[4]+"\tcorrect value: "+argValue[4]);
		System.out.println("output error[5]: "+error[5]+"\tcorrect value: "+argValue[5]);
		*/return error;
	}

	public void save(File argFile) throws IOException {
		//File weightsfile = new File("weightsfile.lqn.txt");
		if(argFile.exists())
		{
			FileWriter erasor = new FileWriter(argFile);
			erasor.write(new String());
			erasor.close();
		}
		else
		{
			argFile.createNewFile();
		}
		ArrayList<double[]> weight_vector = new ArrayList<double[]>(0);
		//for input nodes
		for(int i = 0; i<this.input_num; i++)
		{
			double[] buffer;
			buffer = new double[1];//input nodes only have one input
			for(int j=0;j<this.input_nodes.get(i).getWeights().size();j++)
			{
				buffer[j] = this.input_nodes.get(i).getWeights().get(j);
			}
			weight_vector.add(i, buffer);
		}
		if(this.writeWeights(weight_vector, argFile) == 0)
			System.out.println("Writing input nodes fails");
		
		//for hidden nodes
		for(int i = 0; i<this.hidden_num; i++)
		{
			double[] buffer;
			buffer = new double[this.input_num];
			for(int j=0;j<this.hidden_nodes.get(i).getWeights().size();j++)
			{
				buffer[j] = this.hidden_nodes.get(i).getWeights().get(j);
			}
			weight_vector.add(i, buffer);
		}
		if(this.writeWeights(weight_vector,argFile) == 0)
			System.out.println("Writing hidden nodes fails");
		
		//for output nodes
		for(int i = 0; i<this.output_num; i++)
		{
			double[] buffer;
			buffer = new double[this.hidden_num];
			for(int j=0;j<this.output_nodes.get(i).getWeights().size();j++)
			{
				buffer[j] = this.output_nodes.get(i).getWeights().get(j);
			}
			weight_vector.add(i, buffer);
		}
		if(this.writeWeights(weight_vector,argFile) == 0)
			System.out.println("Writing output nodes fails");
		
	}

	public void load(String argFileName) throws IOException {
		File weightsfile = new File(argFileName);
		if(weightsfile.exists())
		{
			if(weightsfile.isFile())
			{
				System.out.println("Found the file, the name is " + weightsfile.getName());
				BufferedReader content = new BufferedReader(new FileReader(weightsfile));
				String str;
				int type = 0;
				int id = 0;
				while((str = content.readLine())!=null)
				{
					if(str.charAt(0) == '=')
					{
						id = 0;
						type++;
					}
					else
						this.readWeights(str, type, id++);
				}
				content.close();
			}
			else
			{
				System.out.println("Not a File");
			}
		}
	}

	public double sigmoid(double x) {
		return 2*Math.pow((Math.exp(x * (-1.0)) + 1), (-1)) - 1;
	}

	public double customSigmoid(double x) {
		return (upper_bound-lower_bound)*Math.pow((Math.exp(x * (-1.0)) + 1), (-1))+lower_bound;
	}

	public void initializeWeights() {
		for(int i = 0; i<this.hidden_num; i++)
		{
			hidden_nodes.get(i).initWeights(true, 0);
		}
		for(int i = 0; i<this.input_num; i++)
		{
			input_nodes.get(i).initWeights(true, 0);
		}
		for(int i = 0; i<this.output_num; i++)
		{
			output_nodes.get(i).initWeights(true, 0);
		}
	}

	public void zeroWeights() {
		for(int i = 0; i<this.hidden_num; i++)
		{
			hidden_nodes.get(i).initWeights(false, 0);
		}
		for(int i = 0; i<this.input_num; i++)
		{
			input_nodes.get(i).initWeights(false, 0);
		}
		for(int i = 0; i<this.output_num; i++)
		{
			output_nodes.get(i).initWeights(false, 0);
		}
	}
	
	public int writeWeights(ArrayList<double[]> content, File file) throws IOException
	{
		BufferedWriter bw = new BufferedWriter (new FileWriter(file,true));
		for(int i=0; i<content.size(); i++)
		{
			for(int j = 0; j<content.get(i).length;j++)
			{
				bw.write(content.get(i)[j] + ",");
				bw.flush();
			}
			bw.newLine();
			bw.flush();
		}
		bw.write("============================================");
		bw.newLine();
		bw.flush();
		bw.close();
		content.clear();
		System.out.println("Writing successed");
		return 1;
	}
	
	public void readWeights(String str, int type, int id)
	{
		//input node:0 hidden node:1 output nodes:2
		//input node
		int end_index = 0;
		if(type == 0)
		{
			int count = 0;
			double[] weights_array;
			weights_array = new double[1];
			//ArrayList<Character> weights_arraylist = new ArrayList<Character>(0);
			
			for(int i = 0; i<str.length(); i++)
			{
				if(str.charAt(i) == ',')
				{
					//String buffer = str.substring(end_index, i);
					//System.out.println(Double.parseDouble(buffer));
					weights_array[count] = Double.parseDouble(str.substring(end_index, i));
					count++;
					end_index = i+1;
				}
			}
			
			this.input_nodes.get(id).setWeights(weights_array);
		}
		
		//hidden node
		else if(type == 1)
		{
			int count = 0;
			double[] weights_array;
			weights_array = new double[this.input_num];
			//ArrayList<Character> weights_arraylist = new ArrayList<Character>(0);
			
			for(int i = 0; i<str.length(); i++)
			{
				if(str.charAt(i) == ',')
				{
					//String buffer = str.substring(end_index, i);
					//System.out.println(Double.parseDouble(buffer));
					weights_array[count] = Double.parseDouble(str.substring(end_index, i));
					count++;
					end_index = i+1;
				}
			}
			this.hidden_nodes.get(id).setWeights(weights_array);
		}
		
		//output node
		else if(type == 2)
		{
			int count = 0;
			double[] weights_array;
			weights_array = new double[this.hidden_num];
			//ArrayList<Character> weights_arraylist = new ArrayList<Character>(0);
			
			for(int i = 0; i<str.length(); i++)
			{
				if(str.charAt(i) == ',')
				{
					//String buffer = str.substring(end_index, i);
					//System.out.println(Double.parseDouble(buffer));
					weights_array[count] = Double.parseDouble(str.substring(end_index, i));
					count++;
					end_index = i+1;
				}
			}
			this.output_nodes.get(id).setWeights(weights_array);
		}
		else
		{
			System.out.println("The layer number is wrong here, \"type\" should not be larger than 2");
		}
	}

	public void setBound(double upper, double lower)
	{
		for(int i=0; i<this.input_num; i++)
			this.input_nodes.get(i).setBound(upper, lower);
		for(int i=0; i<this.hidden_num; i++)
			this.hidden_nodes.get(i).setBound(upper, lower);
		for(int i=0; i<this.output_num; i++)
			this.output_nodes.get(i).setBound(upper, lower);
	}

	public double sigmoid_of_cells(double x)
	{
		return this.hidden_nodes.get(0).sigmoid(x);
	}

	public double getCorrectOutput()
	{
		return this.output_nodes.get(0).getCorrectOutput();
	}
	
	public void setStep(double step)
	{
		for(int i=0; i<this.input_num; i++)
			this.input_nodes.get(i).p_step_size = step;
		for(int i=0; i<this.hidden_num; i++)
			this.hidden_nodes.get(i).p_step_size = step;
		for(int i=0; i<this.output_num; i++)
			this.output_nodes.get(i).p_step_size = step;
	}
	
	public void setMomentum(double x)
	{
		for(int i=0; i<this.input_num; i++)
			this.input_nodes.get(i).setMomentum(x);
		for(int i=0; i<this.hidden_num; i++)
			this.hidden_nodes.get(i).setMomentum(x);
		for(int i=0; i<this.output_num; i++)
			this.output_nodes.get(i).setMomentum(x);
	}

	//====================================================================================================================================================
	/*public double[] training_binary_four_times(NeuralNetwork binary_network)
	{
		binary_network.setBound(1, 0);
		double[] error;
		error = new double[5];
		error[0] = 10.0;
		error[1] = 10.0;
		error[2] = 10.0;
		error[3] = 10.0;
		error[4] = 10.0;
		
		double[] input1 = {0.0,1.0};
		double[] input2 = {1.0,0.0};
		double[] input3 = {0.0,0.0};
		double[] input4 = {1.0,1.0};
		double[] input = {0.0,0.0};
		{
			input[0] = Math.round((float)Math.random());
			input[1] = Math.round((float)Math.random());
			
			//error[0] = Math.abs(binary_network.train(input, (double)((int)input[0]^(int)input[1])));
			error[1] = Math.abs(binary_network.train(input1, (double)((int)input1[0]^(int)input1[1])));
			error[3] = Math.abs(binary_network.train(input3, (double)((int)input3[0]^(int)input3[1])));
			error[2] = Math.abs(binary_network.train(input2, (double)((int)input2[0]^(int)input2[1])));
			//error[3] = Math.abs(binary_network.train(input3, (double)((int)input3[0]^(int)input3[1])));
			error[4] = Math.abs(binary_network.train(input4, (double)((int)input4[0]^(int)input4[1])));
		}
		return error;
	}

	public double[] training_bipolar_four_times(NeuralNetwork bipolar_network)
	{
		
		double[] bpinput1 = {-1.0,1.0};
		double[] bpinput2 = {1.0,-1.0};
		double[] bpinput3 = {-1.0,-1.0};
		double[] bpinput4 = {1.0,1.0};
		double[] bpinput = {1,0};
		double[] bpinput_vector = {-1.0, -1.0};
		double[] error;
		error = new double[5];
		error[0] = 10.0;
		error[1] = 10.0;
		error[2] = 10.0;
		error[3] = 10.0;
		error[4] = 10.0;
		{
			bpinput[0] = Math.round((float)Math.random());
			bpinput[1] = Math.round((float)Math.random());
			bpinput_vector[0] = bpinput[0]*2 -1;
			bpinput_vector[1] = bpinput[1]*2 -1;
			double result = (double)(((int)bpinput[0] ^ (int)bpinput[1])*2 -1);
			//error[0] = Math.abs(bipolar_network.train(bpinput_vector, result));

			error[1] = Math.abs(bipolar_network.train(bpinput1, (double)(1)));
			error[3] = Math.abs(bipolar_network.train(bpinput3, (double)(-1)));
			error[2] = Math.abs(bipolar_network.train(bpinput2, (double)(1)));
			//error[3] = Math.abs(bipolar_network.train(bpinput3, (double)(-1)));
			error[4] = Math.abs(bipolar_network.train(bpinput4, (double)(-1)));
			//error[1] = Math.abs(bipolar_network.train(bpinput1, (double)(1)));
			System.out.println("bipolar: "+error[0]+"\t"+error[1]+"\t"+error[2]+"\t"+error[3]+"\t"+error[4]);
			//System.out.println("bipolar: "+(error[0]+error[1]+error[2]+error[3]+error[4]));
		}
		return error;
	}
	
	public ArrayList<Double> training_binary_until_threshold(NeuralNetwork binary_network)
	{
		ArrayList<Double> error_dot = new ArrayList<Double>(0);
		binary_network.setBound(1, 0);
		//binary_network.setMomentum(0.9);
		int epoch_count = 0;
		while(true)
		{
			double[] error;
			error = new double[5];
			error = training_binary_four_times(binary_network);
			error_dot.add(error[1]+error[2]+error[3]+error[4]);
			epoch_count++;
			if((error[1]+error[2]+error[3]+error[4]) < THRESHOLD )
			{
				//System.out.println("Binary: Epoch: "+(int)(error_dot.size()*1.5));
				break;
			}
			if(epoch_count == TIMEOUT_LOOP_COUNT )
			{
				System.out.println("Binary: TIMEOUT");
				error_dot.add(-1.0);
				break;
			}	
			
		}
		return error_dot;
	}

	public ArrayList<Double> training_binary_until_threshold()
	{
		return this.training_binary_until_threshold(this);
	}

	public ArrayList<Double> training_bipolar_until_threshod(NeuralNetwork bipolar_network)
	{
		ArrayList<Double> error_dot = new ArrayList<Double>(0);
		bipolar_network.setBound(1, -1);
		//bipolar_network.setMomentum(0.9);
		int epoch_count = 0;
		
		while(true)
		{
			
			double[] error;
			error = new double[5];
			error = training_bipolar_four_times(bipolar_network);
			error_dot.add(error[1]+error[2]+error[3]+error[4]);
			epoch_count++;
			if((error[1]+error[2]+error[3]+error[4]) < THRESHOLD )
			{
				//System.out.println("Bipolar: Epoch: "+(int)(error_dot.size()*6.0/4.0));
				break;
			}
			if(epoch_count == TIMEOUT_LOOP_COUNT )
			{
				System.out.println("Bipolar: TIMEOUT");
				error_dot.add(-1.0);
				File timeoutweights = new File("timeout.txt");
				break;
			}	
			
		}
		return error_dot;
	}

	public ArrayList<Double> training_bipolar_until_threshod()
	{
		return this.training_bipolar_until_threshod(this);
	}*/
	//===============================================================================================================================================
	/*public double[] training_bipolar_mega_four_times(NeuralNetwork bipolar_network)
	{
		
		double[] bpinput1 = {-1000000.0,1.0};
		double[] bpinput2 = {1.0,-1000000.0};
		double[] bpinput3 = {-1000000.0,-1000000.0};
		double[] bpinput4 = {1.0,1.0};
		double[] bpinput = {1,0};
		double[] bpinput_vector = {-1000000.0, -1000000.0};
		double[] error;
		error = new double[5];
		error[0] = 10.0;
		error[1] = 10.0;
		error[2] = 10.0;
		error[3] = 10.0;
		error[4] = 10.0;
		{
			bpinput[0] = Math.round((float)Math.random());
			bpinput[1] = Math.round((float)Math.random());
			if(bpinput[0] == 1) bpinput_vector[0] = bpinput[0];	else bpinput_vector[0] = -1000000.0;
			if(bpinput[1] == 1) bpinput_vector[1] = bpinput[1];	else bpinput_vector[1] = -1000000.0;
			double result = (double)(((int)bpinput[0] ^ (int)bpinput[1])*2 -1);
			error[0] = Math.abs(bipolar_network.train(bpinput_vector, result));

			error[1] = Math.abs(bipolar_network.train(bpinput1, (double)(1)));
			//error[3] = Math.abs(bipolar_network.train(bpinput3, (double)(-1)));
			error[2] = Math.abs(bipolar_network.train(bpinput2, (double)(1)));
			error[3] = Math.abs(bipolar_network.train(bpinput3, (double)(-1)));
			error[4] = Math.abs(bipolar_network.train(bpinput4, (double)(-1)));
			error[1] = Math.abs(bipolar_network.train(bpinput1, (double)(1)));
			//System.out.println("bipolar: "+error[0]+"\t"+error[1]+"\t"+error[2]+"\t"+error[3]+"\t"+error[4]);
		}
		return error;
	}

	public ArrayList<Double> training_bipolar_mega_until_threshod(NeuralNetwork bipolar_network)
	{
		ArrayList<Double> error_dot = new ArrayList<Double>(0);
		bipolar_network.setBound(1, -1);
		//bipolar_network.setMomentum(0.9);
		int epoch_count = 0;
		
		while(true)
		{
			
			double[] error;
			error = new double[5];
			error = training_bipolar_mega_four_times(bipolar_network);
			error_dot.add(error[1]+error[2]+error[3]+error[4]);
			epoch_count++;
			if((error[1]+error[2]+error[3]+error[4]) < THRESHOLD )
			{
				//System.out.println("Mega bipolar: Epoch: "+(int)(error_dot.size()*6.0/4.0));
				break;
			}
			if(epoch_count >= TIMEOUT_LOOP_COUNT*10 )
			{
				//System.out.println("Mega bipolar: TIMEOUT");
				error_dot.add(-1.0);
				File timeoutweights = new File("timeout.txt");
				break;
			}	
			
		}
		return error_dot;
	}

	public ArrayList<Double> training_bipolar_mega_until_threshod()
	{
		return this.training_bipolar_mega_until_threshod(this);
	}*/
	//===============================================================================================================================================
}
