This program runs trains an RL agent that generates sorting networks.

In the begning of the file you can modify the following configurations:
<ul>
<li><b>arraySize</b> - the size of input array for sorting netwok<br /></li>
<li><b>setSize</b> - the number of test arrays for each environment <br /></li>
<li><b>maxSortStages</b> - maximum sorting stages\levels of the sorting network<br /></li>
<li><b>num_of_envs</b> - number of training environments to train the model on<br /></li>
<li><b>max_num_of_comps</b> - maximum number of comparators in the network<br /></li>
<li><b>max_modifications</b> - maximum number of steps in each episode<br /></li>
<li><b>learning_timesteps</b> - number of learning timesteps in the learning process<br /></li>
<li><b>inversions_threshold</b> - only include arrays with inversions above this number<br /></li>
</ul>
<br />
On each episode of learning the following details are printed:
<ul>
  <li><b>previous iteration sorted</b> - number of arrays sorted by previously generated network (only first environment in the vectorized env)<br /></li>
  <li><b>current iteration iteration sorted</b> - number of arrays sorted by current generated network (only first environment in the vectorized env)<br /></li>
  <li><b>sorted vectors lost since previous iteration</b> - lost vectors between current and previous network<br /></li>
  <li><b>current best score</b> - the best current score<br />
	<li><b>current best network</b> - the current smallest perfect network (if exists)<br />
  <li><b>overall perfect networks generated</b> - overall number of perfect networks generated in all sessions <br /></li>
  <li><b>smallest perfect network</b> - size - size (number of comparators) of the smallest perfect network<br /></li>
  <li><b>smallest perfect network</b> - depth - depth of the smallest perfect network.<br /></li>
</ul>
At the end of each training session, the agent will run and the following will be printed:
<ul>
  <li>on environment <b><i>x</i></b> , reward <b><i>y</i></b>, sorted <b><i>z</i></b> - the agent ran on sub-enviornment <b><i>x</i></b> and generated a network which sorted <b><i>z</i></b> arrays and got the reward <b><i>y</i></b> </li>
  <li>if the agent generated a perfect network, it'll output "Found perfect network after x steps" and the network will be printed  <br /></li>
</ul>

Expected output: <br />
The initial configurations are pretty loose and a perfect (rather large) network should be generated after the first or second training sessions (about 5 to 10 minutes). <br />
Smaller "maxSortStages" and larger "arraySize" configurations will be more difficult for the agent and will take more time (even hours or days). <br />
<b> Since the current implementation initilizes the training sets randomly, results regarding runtime may vary between runs </b> <br />
<b> For example: setting maxSortStages to 16 for arraySize to 8 can take a couple of hours to a day in order to generate networks smaller than 30 </b> 
<br />
Using the tester:<br />
<i> python network_tester.py 1,2,3,4,0,3,2,1 true </i> </br>
Change from true to false to hide details of the sorting process
