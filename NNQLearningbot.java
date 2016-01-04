import java.awt.*;   
import java.awt.geom.*;   
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintStream;
import Sarb.NeuralNet;
import java.io.File;
import java.io.FileWriter;
import java.util.Random;
import java.util.Vector;

import robocode.*;   
       
import robocode.AdvancedRobot;   
       
    public class NNQLearningbot extends AdvancedRobot   
    {   
      public static final double PI = Math.PI;   
      private Target target;   
      private QTable table;   
      private Learner learner;   
      private double reward = 0.0;   
      private double firePower;   
      private int direction = 1;   
      private int isHitWall = 0;   
      private int isHitByBullet = 0;   
      
      double rewardForWin=100;
      double rewardForDeath=-10;
      double accumuReward=0.0;
  	
    //Neural network
  	private static NeuralNet[] network = new NeuralNet[Action.NumRobotActions];
  	private static final double rho = 0.1; //learning rate
  	private static final double momentum = 0.0;
  	private static final double upper = 36;
  	private static final double lower = 0;
  	private static final int hidden_num = 4;
  	private static final double NN_alpha = 0.1; //learning rate for NN Q learning
  	private static final double NN_lambda = 0.9; //Discount rate for NN Q learning
  	private static double NN_epsilon = 0.0;
  	private int NN_last_action = 0;
  	private double[] NN_last_states = new double[5]; 
  	
  	//Whether do the space reduction or not. For Q learning, space reduction is needed while for NN, there is no need for space reduction.
  	public static boolean spaceReduction=false;
  	
  	static
  	{
  		for(int i=0; i<Action.NumRobotActions; i++)
  		{
  			network[i] = new NeuralNet(5, hidden_num, rho, momentum, lower, upper, true);  //6 neural network for training 6 different actions
  		} 
  	}
  	
  	//For statistics
  	private static int episodeNum = 100;
  	private static int episode = 0;
  	private static int wins = 0;
  	private static Vector<Double> winrates = new Vector<Double>(0);
  	
  	//e(s) statistics
  	private static double total_error_in_one_round = 0.0;
  	private static Vector<Double> error_statistics = new Vector<Double>(0);
  	
  	//Q-value learning history
  	private static Vector<Double> Q_value_history = new Vector<Double>(0);
  	private static final double test_heading = 100;
	private static final double test_target_distance = 100;
	private static final double test_target_bearing = 70;
	private static final int test_action = 0;
	private static final int test_hitWall = 0;
	private static final int test_hitByBullet = 0;
	private static double NN_test_heading = test_heading/180 -1;
	private static double NN_test_target_distance = test_target_distance/500 -1;
	private static double NN_test_target_bearing = test_target_bearing/180-1;
	private static final double NN_test_hitWall = 0;
	private static final double NN_test_hitByBullet = 0;
	private static int NN_test_action = test_action;
      
      public void run()   
      {   
    	if(spaceReduction && this.getNumRounds() - this.getRoundNum() == 1)
        {
    		try{
    			this.saveWinRates(winrates, new String("LUT_winrates.txt"));
				this.saveQ(Q_value_history, new String("Q_history.txt"));
    			
    		} 
    		catch (IOException e) {
				e.printStackTrace();
				System.out.println("SAVE WINRATES/Q_HISTORY FAILED");
			}
        }
    	else if(this.getNumRounds() - this.getRoundNum() == 1 && !spaceReduction)
    	{
    		try
    		{
    			this.saveWinRates(winrates, new String("NN_winRates.txt"));
    			this.saveError(error_statistics, new String("NN_error.txt"));
    			this.saveQ(Q_value_history, new String("Q_history.txt"));
    		}
    		catch(IOException e)
    		{
    			e.printStackTrace();
    			System.out.println("save NN_winRates/NN_error/Q_history file failed");
    		}
    		
    	}
    	this.clearAllEvents();
    	System.out.println("The win rate is"+wins+"/"+episode);
    	
    	System.out.println("Q value in 204 is: "+ QTable.table[34][0]);
    	
    	if(episode == episodeNum)
    	{
    		this.storeWinRate((double)wins/episode);
    		episode = 1;
    		wins = 0;
    	}
    	else
    	{
    		episode++;
    	}
    	
    	if(spaceReduction==false)  //For NN, the last states is initialized to zero at the begining of each round.
    	{
    		for(int i=0; i<5; i++)
    		{
    			NN_last_states[i]=0.0;
    		}
    		
    		error_statistics.add(this.total_error_in_one_round);
    		this.total_error_in_one_round = 0.0;
    		
    	}
    	
    	//record Q value
    	if(spaceReduction==true)
    	{
    		int stateForLUT = this.getStateForLUT();
    		Q_value_history.add(QTable.table[stateForLUT][test_action]);
    	}
    	else
    	{
    		double[] temp = new double[5];
    		temp[0] = NN_test_heading;
    		temp[1] = NN_test_target_distance;
    		temp[2] = NN_test_target_bearing;
    		temp[3] = NN_test_hitWall;
    		temp[4] = NN_test_hitByBullet;
    		
    		Q_value_history.add(network[NN_test_action].outputFor(temp));
    	}
    	
    	target.distance = 100000;   
       
    	setColors(Color.green, Color.white, Color.green);   
    	setAdjustGunForRobotTurn(true);   
    	setAdjustRadarForGunTurn(true);   
    	turnRadarRightRadians(2 * PI);   
    
      }   
 
      public void actualRun()
      {
    	  table = new QTable();   
          loadData();   
  		  learner = new Learner(table);   
  		  target = new Target(); 
  		
    	  if(getRoundNum()>300)
      		{
      			learner.explorationRate=0.1;
      		}
      		
          while (true)   
          {   
            robotMovement();   
            firePower = 400/target.distance;   
            if (firePower > 3)   
              firePower = 3;   
            radarMovement();   
            gunMovement();   
            if (getGunHeat() == 0) {   
              setFire(firePower);   
            }   
            execute();   
          }   
      }
     
      private void robotMovement()   
      {    
    	  
        int action; 
        
        if(spaceReduction==true)
        {
        	int state = getStateForLUT();  
        	action = learner.selectAction(state);    
        	learner.learn(state, action, reward); 
            //learner.learnSARSA(state, action, reward);
        }
        else
        {
        	action = this.NeuralNetforAction(getHeading()/180-1,target.distance/500-1,target.bearing/180-1,isHitWall,isHitByBullet, reward);
        }
     
        accumuReward+=reward;
        reward = 0.0;   
        isHitWall = 0;   
        isHitByBullet = 0;   
       
        switch (action)   
        {   
          case Action.RobotAhead:   
            setAhead(Action.RobotMoveDistance);   
            break;   
          case Action.RobotBack:   
            setBack(Action.RobotMoveDistance);   
            break;   
          case Action.RobotAheadTurnLeft:   
            setAhead(Action.RobotMoveDistance);   
            setTurnLeft(Action.RobotTurnDegree);      
            break;   
          case Action.RobotAheadTurnRight:   
            setAhead(Action.RobotMoveDistance);   
            setTurnRight(Action.RobotTurnDegree);    
            break;   
          case Action.RobotBackTurnLeft:   
            setAhead(Action.RobotMoveDistance);   
            setTurnRight(Action.RobotTurnDegree);    
            break;   
          case Action.RobotBackTurnRight:   
            setAhead(target.bearing);   
            setTurnLeft(Action.RobotTurnDegree);      
            break;   
        }   
      }    
      
      //get test state for look up table, don't use for neural network
      private int getTestStateForLUT()   
      {    
    	  int heading =State.getHeading(test_heading);   
          int targetDistance = State.getTargetDistance(test_target_distance);   
          int targetBearing =(int)State.getTargetBearing(test_target_bearing);      
          int state = State.Mapping[heading][targetDistance][targetBearing][test_hitWall][test_hitByBullet];   
          return state;   
      }   
      
      //get state for look up table, don't use for neural network
      private int getStateForLUT()   
      {   
        int heading = State.getHeading(getHeading());   
        int targetDistance = State.getTargetDistance(target.distance);   
        int targetBearing = State.getTargetBearing(target.bearing);   
        out.println("State(" + heading + ", " + targetDistance + ", " + targetBearing + ", " + isHitWall + ", " + isHitByBullet + ")");   
        int state = State.Mapping[heading][targetDistance][targetBearing][isHitWall][isHitByBullet];   
        return state;   
      }   
       
      private void radarMovement()   
      {   
        double radarOffset;   
        if (getTime() - target.ctime > 4) {  
          radarOffset = 4*PI;                
        } else {   
       
          //next is the amount we need to rotate the radar by to scan where the target is now   
          radarOffset = getRadarHeadingRadians() - (Math.PI/2 - Math.atan2(target.y - getY(),target.x - getX()));   
          //this adds or subtracts small amounts from the bearing for the radar to produce the wobbling   
          //and make sure we don't lose the target   
          radarOffset = NormaliseBearing(radarOffset);   
          if (radarOffset < 0)   
            radarOffset -= PI/10;   
          else   
            radarOffset += PI/10; 

        }   
        //turn the radar   
        setTurnRadarLeftRadians(radarOffset);   
      }   
       
      private void gunMovement()   
      {   
        long time;   
        long nextTime;   
        Point2D.Double p;   
        p = new Point2D.Double(target.x, target.y);   
        for (int i = 0; i < 20; i++)   
        {   
          nextTime = (int)Math.round((getrange(getX(),getY(),p.x,p.y)/(20-(3*firePower))));   
          time = getTime() + nextTime - 10;   
          p = target.guessPosition(time);   
        }   
        //offsets the gun by the angle to the next shot based on linear targeting provided by the enemy class   
        double gunOffset = getGunHeadingRadians() - (Math.PI/2 - Math.atan2(p.y - getY(),p.x -  getX()));   
        setTurnGunLeftRadians(NormaliseBearing(gunOffset));   
      }   

      //bearing is within the -pi to pi range   
      double NormaliseBearing(double ang){

        if (ang > PI)   
          ang -= 2*PI;   
        if (ang < -PI)    
          ang += 2*PI;   
        return ang;   
      }   
       
      //heading within the 0 to 2pi range   
      double NormaliseHeading(double ang) {   
        if (ang > 2*PI)   
          ang -= 2*PI;   
        if (ang < 0)   
          ang += 2*PI;   
        return ang;   
      }   
       
      //returns the distance between two x,y coordinates   
      public double getrange( double x1,double y1, double x2,double y2 )   
      {   
        double xo = x2-x1;   
        double yo = y2-y1;   
        double h = Math.sqrt( xo*xo + yo*yo );   
        return h;   
      }   
       
      //gets the absolute bearing between to x,y coordinates   
      public double absbearing( double x1,double y1, double x2,double y2 )   
      {  
        double xo = x2-x1;   
        double yo = y2-y1;   
        double h = getrange( x1,y1, x2,y2 );   
        if( xo > 0 && yo > 0 )   
        {   
          return Math.asin( xo / h );   
        }   
        if( xo > 0 && yo < 0 )   
        {   
          return Math.PI - Math.asin( xo / h );   
        }   
        if( xo < 0 && yo < 0 )   
        {   
          return Math.PI + Math.asin( -xo / h );   
        }   
        if( xo < 0 && yo > 0 )   
        {   
          return 2.0*Math.PI - Math.asin( -xo / h );   
        }   
        return 0;   
      }   
       
      public void onBulletHit(BulletHitEvent e)   
      {  

        if (target.name == e.getName())   
        {     
          double change = e.getBullet().getPower() * 9;   
          out.println("Bullet Hit: " + change);   
          reward += change;   
        }   
      }   
       

      public void onBulletMissed(BulletMissedEvent e)   
      {   
        double change = -e.getBullet().getPower();   
        out.println("Bullet Missed: " + change);   
       
		reward += change;   
      }   
       
      public void onHitByBullet(HitByBulletEvent e)   
      {   
        if (target.name == e.getName())   
        {   
          double power = e.getBullet().getPower();   
          double change = -(4 * power + 2 * (power - 1));   
          out.println("Hit By Bullet: " + change);   
          reward += change;   
        }   
        isHitByBullet = 1;   
      }   
       
      public void onHitRobot(HitRobotEvent e)   
      {   
        if (target.name == e.getName())   
        {   
          double change = -6.0;   
          out.println("Hit Robot: " + change);   
          reward += change;   
        }   
      }   
       
      public void onHitWall(HitWallEvent e)   
      {   
           
        double change = -(Math.abs(getVelocity()) * 0.5 - 1);   
        out.println("Hit Wall: " + change);   
        reward += change;   
        isHitWall = 1;   
      }   

      public void onScannedRobot(ScannedRobotEvent e)   
      {   
        if ((e.getDistance() < target.distance)||(target.name == e.getName()))   
        {   
          //the next line gets the absolute bearing to the point where the bot is   
          double absbearing_rad = (getHeadingRadians()+e.getBearingRadians())%(2*PI);   
          //this section sets all the information about our target   
          target.name = e.getName();   
          double h = NormaliseBearing(e.getHeadingRadians() - target.head);   
          h = h/(getTime() - target.ctime);   
          target.changehead = h;   
          target.x = getX()+Math.sin(absbearing_rad)*e.getDistance(); //works out the x coordinate of where the target is   
          target.y = getY()+Math.cos(absbearing_rad)*e.getDistance(); //works out the y coordinate of where the target is   
          target.bearing = e.getBearingRadians();   
          target.head = e.getHeadingRadians();   
          target.ctime = getTime();             //game time at which this scan was produced   
          target.speed = e.getVelocity();   
          target.distance = e.getDistance();   
          target.energy = e.getEnergy();   
        }   
      }   
       
      public void onRobotDeath(RobotDeathEvent e)   
      {   
       
        if (e.getName() == target.name)   
        {
        	target.distance = 10000; 
        }
         
      }   
       
      public void onWin(WinEvent event)   
      {   
          
        reward+=rewardForWin;
        robotMovement();
        saveData(); 
		 int winningFlag=7;

		 PrintStream w = null; 
		    try 
		    { 
		      w = new PrintStream(new RobocodeFileOutputStream("/home/lili/workspace/EECE592/BPRL/src/QJan1survival0.2.xlsx", true)); 
		      w.println(accumuReward+" "+getRoundNum()+"\t"+winningFlag+" "+" "+learner.explorationRate); 
		      if (w.checkError()) 
		        System.out.println("Could not save the data!");  
		      w.close(); 
		    } 
		    catch (IOException e) 
		    { 
		      System.out.println("IOException trying to write: " + e); 
		    } 
		    finally 
		    { 
		      try 
		      { 
		        if (w != null) 
		          w.close(); 
		      } 
		      catch (Exception e) 
		      { 
		        System.out.println("Exception trying to close witer: " + e); 
		      } 

		    } 
      }   
       
      public void onDeath(DeathEvent event)   
      {   
    	reward+=rewardForDeath;
      	robotMovement();
        saveData();   
         
        int losingFlag=5;
		PrintStream w = null; 
		    try 
		    { 
		      w = new PrintStream(new RobocodeFileOutputStream("/home/lili/workspace/EECE592/BPRL/src/QJan1survival0.2.xlsx", true)); 
		      w.println(accumuReward+" "+getRoundNum()+"\t"+losingFlag+" "+" "+learner.explorationRate); 
		      if (w.checkError()) 
		        System.out.println("Could not save the data!"); 
		      w.close(); 
		    } 
		    catch (IOException e) 
		    { 
		      System.out.println("IOException trying to write: " + e); 
		    } 
		    finally 
		    { 
		      try 
		      { 
		        if (w != null) 
		          w.close(); 
		      } 
		      catch (Exception e) 
		      { 
		        System.out.println("Exception trying to close witer: " + e); 
		      } 

		    } 
      }   
       
      public void loadData()   
      {   
        try   
        {   
          table.loadData(getDataFile("LUTableJan1.dat"));   
        }   
        catch (Exception e)   
        {   
        }   
      }   
       
      public void saveData()   
      {   
        try   
        {   
          table.saveData(getDataFile("LUTableJan1.dat"));   
        }   
        catch (Exception e)   
        {   
          out.println("Exception trying to write: " + e);   
        }   
      }   
      
      /**
  	 * Using Neural network to determine action
  	 * @param heading my heading, (-1,+1)
  	 * @param target distance target distance, (-1,+1)
  	 * @param bearing relative bearing, (-1,+1)
  	 * @param reward the reward from last turn, double
  	 * @return the action should execute
  	 */
      public int NeuralNetforAction(double heading, double targetDistance, double targetBearing, double isHitWall, double isHitByBullet, double reward)
      {
    	  int action = 0;
    	  double[]  NN_current_states= new double[5];
    	  NN_current_states[0]=heading;
    	  NN_current_states[1]=targetDistance;
    	  NN_current_states[2]=targetBearing;
    	  NN_current_states[3]=isHitWall;
    	  NN_current_states[4]=isHitByBullet;
    	  
    	  //get the best action
    	  for(int i=0; i<Action.NumRobotActions; i++)
    	  {
    		  if(network[i].outputFor(NN_current_states)>network[action].outputFor(NN_current_states))
    		  {
    			  action = i;
    		  }
    	  }
    	  
    	  //update weights
    	  double NN_Q_new=network[action].outputFor(NN_current_states);
    	  double error_signal = 0;
    	  if(NN_last_states[0]==0.0||NN_last_states[1]==0.0);
    	  else
    	  {
    		  error_signal = NN_alpha*(reward + NN_lambda * NN_Q_new - network[NN_last_action].outputFor(NN_last_states));
    	  }
    	  
    	  this.total_error_in_one_round += error_signal*error_signal/2;
    	  double correct_old_Q = network[NN_last_action].outputFor(NN_last_states) + error_signal;
    	  network[NN_last_action].train(NN_last_states, correct_old_Q);
    	  
    	  if(Math.random() < NN_epsilon)
    	  {
    		  action = new Random().nextInt(Action.NumRobotActions);
    	  }
    	  
    	  for(int i=0; i<5; i++)
    	  {
    		  NN_last_states[i] = NN_current_states[i];
    	  }
    	  
    	  NN_last_action=action;
    	  return action;
      }
      
      public void saveError(Vector<Double> error_statistics, String filename) throws IOException
      {
    	  File errorFile = new File(filename);
    	  BufferedWriter bw = new BufferedWriter(new FileWriter(errorFile, false));
    	  for(int i=0; i< error_statistics.size(); i++)
    	  {
    		  bw.write(error_statistics.get(i)+" ");
    		  bw.newLine();
    		  bw.flush();
    	  }
    	  bw.close();
      }
      
      public void saveQ(Vector<Double> Q_history, String filename) throws IOException
      {
    	  File QFile = new File(filename);
    	  BufferedWriter bw = new BufferedWriter(new FileWriter(QFile, false));
    	  for(int i=0; i<Q_history.size(); i++)
    	  {
    		  bw.write(Q_history.get(i)+" ");
    		  bw.newLine();
    		  bw.flush();
    	  }
    	  bw.close();
      }
      
      public void storeWinRate(double winRate)
      {
    	  winrates.add(winRate);
    	  for(int i=0; i<winrates.size(); i++)
    	  {
    		  if(winRate<winrates.get(i))
    		  {
    			  break;
    		  }
    		  else if(i==winrates.size()-1)
    		  {
    			  if(spaceReduction)
    			  {
    				  this.saveData();
    				  
    			  }
    			  else
    			  {
    				  for(int j=0; j<Action.NumRobotActions; j++)
    				  {
    					  network[j].save(new File("NN_best_weights_network("+j+").txt"));
    				  }
    			  }
    		  }
    	  }
      }
      
      public void saveWinRates(Vector<Double> winRates, String filename) throws IOException
      {
    	  File winRatesFile = new File(filename);
    	  BufferedWriter bw = new BufferedWriter(new FileWriter(winRatesFile, false));
    	  for(int i=0; i<winRates.size(); i++)
    	  {
    		  bw.write(winRates.get(i)+" ");
    		  bw.newLine();
    		  bw.flush();
    	  }
    	  bw.close();  
      }
     
   }  

