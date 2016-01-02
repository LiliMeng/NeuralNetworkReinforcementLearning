package part1_NN;

import java.util.ArrayList;

public class InputCell extends NeuralCell{

	private double value = 0;
	private double weight = 0;
	private double previous_weight_incremental = 0.0;
	private boolean constant_weight = false;
	/**
	 * Constructor for InputCell, it can be constructed with an input value.
	 * @param input_value the input value
	 */
	public InputCell(double input_value) {
		super();
		this.value = input_value;
		this.weight = 1.0;
		this.previous_weight_incremental = 0.0;
		this.constant_weight = true;
	}

	/**
	 * Constructor for InputCell without an input value
	 */
	public InputCell()
	{
		super();
		this.weight = 1.0;//Math.random()-0.5;
		this.value = 0.0;
		this.previous_weight_incremental = 0.0;
		this.constant_weight = true;
	}
	
	/**
	 * Shadow the NeuralCell.updateErrorOfLowerCells(),
	 * input nodes have no lower cells.
	 * Do Nothing
	 */
	public void updateErrorOfLowerCells() {
		return;
	}
	/**
	 * Set the input value for input nodes.
	 * @param input_value input value
	 */
	public void setInput(double input_value)
	{
		this.value = input_value;
	}
	/**
	 * Shadow the NeuralCell.getOutput() because InputCells' weight
	 * is scalar instead of vector.
	 */
	public double getOutput()
	{
		//if(this.constant_weight)	return this.value;
		//return super.sigmoid(this.value*this.weight);
		return this.value;
	}
	/**
	 * Shadow the NeuralCell.getSumFromInput().
	 * InputCell doesn't have lower cells
	 */
	public void getSumFromInput(){}
	/**
	 * InputCell's weight is a scalar
	 */
	public void updateWeights()
	{
		return;
		//if(this.constant_weight)	return;
		//double delta;
		//delta = this.getError();
		//double incremental = delta * this.value * this.p_step_size + super.getMomentum() * this.previous_weight_incremental;
		//this.weight = this.weight + delta * this.value * this.p_step_size;
		//this.previous_weight_incremental = incremental;
	}
	/**
	 * Shadow the NeuralCell.initWeights(), due to scalar weight
	 */
	public void initWeights(boolean random, double initval)
	{
		if(this.constant_weight)	return;
		if(random)
		{
			this.weight = Math.random()-0.5;
		}
		else
		{
			this.weight = initval;
		}
	}
	/**
	 * Get the weight of this input node (scalar instead of vector)
	 * @return still return an ArrayList, only one element in it
	 */
	public ArrayList<Double> getWeights()
	{
		ArrayList<Double> result= new ArrayList<Double>(0);
		result.add(this.weight);
		return result;
	}
	/**
	 * Set the weight of this input node
	 */
	public int setWeights(double[] setweights)
	{
		if(this.constant_weight)	return 1;		
		this.weight = setweights[0];
		return 1;
	}
}
