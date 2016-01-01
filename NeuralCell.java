package part1_NN;

import java.lang.Math;
import java.util.ArrayList;
import java.util.Vector;

public class NeuralCell {
	static private int lastCellID = -1;
	protected double sigmoid_upper_bound = 1;
	protected double sigmoid_lower_bound = 0;
	//private double weight = Math.random() - 0.5;
	private int cellID = 0;
	private double momentum = 0.0;
	private double output = 0.0;
	protected double sum_from_input = 0.0;
	private ArrayList<NeuralCell> lowerCells = new ArrayList<NeuralCell>(0);
	private ArrayList<Double> weights = new ArrayList<Double>(0);
	private ArrayList<Double> previous_weights_incremental = new ArrayList<Double>(0);
	
	protected double p_step_size = 0.2;
	
	protected double error = 0.0;
	
	public NeuralCell(ArrayList<NeuralCell> input_neural_nodes)
	{
		this.lowerCells = input_neural_nodes;
		for(int i = 0; i < lowerCells.size(); i++)
		{
			this.weights.add(Math.random() - 0.5);
			this.previous_weights_incremental.add(0.0);
		}
		this.p_step_size = 0.2;
		this.cellID = ++lastCellID;
	}
	public NeuralCell(){this.cellID = ++lastCellID;}
	
	public NeuralCell(Vector<InputCell> input_nodes)
	{
		for(int i = 0; i<input_nodes.size(); i++)
		{
			this.lowerCells.add(input_nodes.get(i));
			this.weights.add(Math.random()-0.5);
			this.previous_weights_incremental.add(0.0);
		}
		this.p_step_size = 0.2;
		this.cellID = ++lastCellID;
		
	}
	
	public void setInputNodes(ArrayList<NeuralCell> cell_list)
	{
		lowerCells = cell_list;
	}

	/**
	 * Get the output value of a neural cell.
	 * @return the output
	 */
	public double getOutput()
	{
		if(this.sum_from_input != 0.0) this.output = sigmoid(this.sum_from_input);
		return this.output;
		
	}
	
	public void getSumFromInput()
	{
		double result = 0.0;
		for(int i=0; i<lowerCells.size(); i++)
		{
			result = result + lowerCells.get(i).getOutput()*weights.get(i);
		}
		
		this.sum_from_input = result;
	}
	
	/**
	 * Compute the activation function.
	 */
	public double sigmoid(double x)
	{
		return (this.sigmoid_upper_bound - this.sigmoid_lower_bound) * Math.pow((Math.exp(x * (-1.0)) + 1.0), (-1.0)) + this.sigmoid_lower_bound;
	}
	/**
	 * Compute the error propagation delta caused by this cell for lower cells.
	 */
	public void updateErrorOfLowerCells()
	{
		for(int i = 0; i<lowerCells.size(); i++)
		{
			double newerror = this.getError() * weights.get(i);
			lowerCells.get(i).addError(newerror);
		}
	}
	
	public void updateErrorOfLowerCells(double outer_error)
	{
		for(int i = 0; i<lowerCells.size(); i++)
		{
			double newerror = outer_error * weights.get(i);
			lowerCells.get(i).addError(newerror);
		}
	}
	/**
	 * Update the delta of this cell, often called by higher cells.
	 * @param newdelta the delta backpropagates from one higher cell
	 */
	public void addError(double newerror)
	{
		this.error += newerror;

	}
	/**
	 * Update the weights vector.
	 */
	public void updateWeights()
	{
		double delta = 0.0;
		delta = this.getError();
		for(int i=0; i<weights.size(); i++)
		{
			double incremental = p_step_size*delta*lowerCells.get(i).getOutput() + this.getMomentum()*this.previous_weights_incremental.get(i);
			weights.set(i, weights.get(i)+ incremental);
			this.previous_weights_incremental.set(i, incremental);
		}
	}

	public void updateWeights(double delta)
	{
		for(int i=0; i<weights.size(); i++)
		{
			double incremental = p_step_size*delta*lowerCells.get(i).getOutput() + this.getMomentum()*this.previous_weights_incremental.get(i);
			weights.set(i, weights.get(i)+ incremental);
			this.previous_weights_incremental.set(i, incremental);
		}
	}
	/**
	 * Refresh "error", "output" and "sum_from_input", but leave "weights" vector unchanged
	 */
	public void refresh()
	{
		this.error = 0.0;
		this.sum_from_input = 0.0;
		this.output = 0.0;			
	}
	/**
	 * Get the error of this object
	 * @return this.error
	 */
	public double getError()
	{
		return this.error
				* (1.0/(this.sigmoid_upper_bound - this.sigmoid_lower_bound))
				* (this.sigmoid_upper_bound - this.output)
				* (this.output - this.sigmoid_lower_bound);
	}
	/**
	 * get the ID of this object
	 * @return this.cellID
	 */
	public int getID()
	{
		return this.cellID;
	}
	/**
	 * Initialise the weights vector by outer parameters
	 * @param random true:use random initialisation false:use initval
	 * @param initval
	 */
	public void initWeights(boolean random, double initval)
	{
		if(random)
		{
			for(int i = 0; i<weights.size();i++)
			{
				weights.set(i, Math.random()-0.5);
			}
		}
		else
		{
			for(int i = 0; i<weights.size();i++)
			{
				weights.set(i, initval);
			}
		}
	}
	
	/**
	 * Get the weights vector
	 * @return ArrayList<Double> weights
	 */
	public ArrayList<Double> getWeights()
	{
		ArrayList<Double> result= new ArrayList<Double>(0);
		for(int i = 0; i<this.weights.size(); i++)
		{
			result.add(this.weights.get(i));
		}
		return result;
	}
	/**
	 * Set weights vector for this object, used for load weights file
	 * @param setweights the double array for weights
	 * @return 0:fail 1:success
	 */
	public int setWeights(double[] setweights)
	{
		if(setweights.length == this.weights.size());
		else 
		{
			System.out.println("the weights file doesn't match the NN");
			return 0;
		}
		
		for(int i = 0; i<setweights.length; i++)
		{
			this.weights.set(i, setweights[i]);
		}
		return 1;
	}

	public void setBound(double upper, double lower)
	{
		this.sigmoid_upper_bound = upper;
		this.sigmoid_lower_bound = lower;
	}

	public void setMomentum(double x)
	{
		this.momentum = x;
	}

	public double getMomentum()
	{
		return this.momentum;
	}
}
