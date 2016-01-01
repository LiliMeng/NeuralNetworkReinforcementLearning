
/* class used for training LUT
 * 
 */
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import part1_NN.NeuralNetwork;

public class Training_NN {
	
	//Use neural network to replace the look-up table
	//1. Load data from look up table. Convert from one dimensional data to multi dimensions
	//2. Process the table contents into 0, 1 representation
	//3. Set up the neural network, with 6 inputs, and 6 outputs.
	//4. Training, set an error threshold to stop 
	//             count the iterations used, record the error with the number of rounds
	//5. What are the best parameters? 
	
	
	 public static final int NumHeading = 4;  
	 public static final int NumTargetDistance = 10;  
	 public static final int NumTargetBearing = 4;  
	 public static final int NumHitWall = 2;  
	 public static final int NumHitByBullet = 2;  
	 
	 public static final int NumActions = 6;  
	 
	 public static final int NumSpace=NumHeading*NumTargetDistance*NumTargetBearing*NumHitWall*NumHitByBullet*NumActions;
	 
	 //Load look-up table
	 private static double[] table_loaded = new double[NumSpace];
	 
	 //Process the look-up table
	 private static double[] table_processed = new double[NumSpace];
	 
	 //learning rate and momentum
	 public static final double LearningRate = 0.1;   
	 
	 public static final double momentum = 0.0;
	 
	 //threshold
	 public static final double threshold = 1.0;
	 
	 //Initialization
	 private void initialize()   
	 {   
	    for (int i = 0; i < NumSpace; i++)
	    {
	    	table_loaded[i]=0;
	    	table_processed[i]=0;
	    }
	 }   
	 
	 public static void main(String [] args) throws IOException
	 {
		 load_table("table.txt");
		 process_table();
		 NeuralNetwork network = new NeuralNetwork(6,6,12);
		 network.setSetp(learningRate);
		 network.setMomentum(momentum);
		 network.setBound(1.0,0.0);
		 
		 //Training
		 double total_error = 0;
		 double error = 0;
		 double max_error = 0.0;
		 do
		 {
			 total_error = 0.0;
			 for(int i=0; i<NumSpace/NumActions; i++)
			 {
				 error = computeError(network.train(generateInputVector(generateInputandActionFromTable(i*NumActions)), generateCorrectOutput(i*NumActions);
				 total_error += error;
			 }
			 max_error = total_error;
			 System.out.println("total_error:"+ total_error+ " error: "+ error+ " max_error "+ max_error); 
		 }
		 while(total_error > threshold);
		 network.save(new File("network_weights_from_table.txt"));
	 }
	 
	 //compute error
	 public static double computeError(double[] errorFromOutput)
	 {
		 double total_error = 0;
		 for(int i = 0; i < errorFromOutput.length; i++)
		 {
			 total_error += errorFromOutput[i];
		 }
		 
		 return total_error;
	 }
	 
	 //load table
	 public static void load_table(String filename) throws IOException
	 {
		 File readFile = new File(filename);
		 BufferedReader br = new BufferedReader(new FileReader(readFile));
		 String str;
		 int count = 0;
		 while((str = br.readLine())!=null)
		 {
			 if(count < NumSpace)
			 {
				 table_loaded[count]=Double.parseDouble(str);
			 }
			 else
			 {
				 break;
			 }
			 count++;
		 }
		 br.close();
	 }
	 
	 public static void process_table()
	 {
		 for(int i=0; i<NumSpace; i=i+NumActions)
		 {
			 int max = 0;
			 for(int j=0; j<NumActions; j++)
			 {
				 if(table_loaded[i+j]<table_loaded[i+max])
				 {
					 table_processed[i+j] = 0.0;
				 }
				 else
				 {
					 table_processed[i+max] = 0.0;
					 table_processed[i+j] = 1.0;
					 max = j;
				 }
			 }
		 }
	 }
	 
	 //There is an action within return array
	 public static int[] generateInputandActionFromTable(int index)
	 {
		 int heading = index/(NumTargetDistance*NumTargetBearing*NumHitWall*NumHitByBullet*NumActions);
		 int left = index % (NumTargetDistance*NumTargetBearing*NumHitWall*NumHitByBullet*NumActions);
		 int targetDistances = left/(NumTargetBearing*NumHitWall*NumHitByBullet*NumActions);
		 left = left % (NumTargetBearing*NumHitWall*NumHitByBullet*NumActions);
		 int targetBearing = left/(NumHitWall*NumHitByBullet*NumActions);
		 left = left % (NumHitWall*NumHitByBullet*NumActions);
		 int hitWall = left/(NumHitByBullet*NumActions);
		 left = left % (NumHitByBullet*NumActions);
		 int hitByBullet = left % NumActions;
		 int action = left;  //This action might be wrong if index is not times of Action
				 
		 int[] return_array = new int[6];
		 
		 return_array[0]=heading;
		 return_array[1]=targetDistances;
		 return_array[2]=targetBearing;
		 return_array[3]=hitWall;
		 return_array[4]=hitByBullet;
		 return_array[5]=action;
		 
		 return return_array;
	 }
	 
	 public static double[] generateCorrectOutput(int index)
	 {
		 int state_index = index - index%NumActions;
		 double[] correctOutput = new double[NumActions];
		 for(int i=0; i<NumActions; i++)
		 {
			 correctOutput[i]=table_processed[state_index + i];
		 }
		 
		 return correctOutput;
	 }

}
