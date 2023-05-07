Name - Satvik Tyagi

Operating system - Windows 10
IDE - Sublime

Time Travel days used - 3

Instructions:

1.Build and train a network to recognize digits (Task1.py) - This script performs everything outlined in task 1 in project description. In order to test the network on anything the neural network must be trained at least once so we have the pretrained model which we can use. The neural networks are saved in current_dir/results/task1/. 

>> Different option in task1.py:

> --siximg - Output first six images with their ground label
> --scratch_train - Train the NN from scratch
> --cont_train - Continue NN training from a saved model and optimizer
> --epochs_cont - # of epochs for continued training
> --network_diag - Print diagram structure
> --loss_diag - Get loss graph for training
> --test_net - Test the trained network and print results
> --test_net_custom - Test the pre trained network and print results on custom data

>> Command to run in "cmd" - python task1.py --siximg True --scratch_train True.........
>> Pick commands that you want to run and execute the code in the above format
>> By defualt all commands are set to False.


2. Analyzing network and transer learning (Task2.py) - This script performs everything outlined in task 2 and task 3 in project description. This makes use of the pretrained model from task 1 to execute itself. Everthing else is set up, the program can be directly run. The neural networks are saved in current_dir/results/task2/.

3. Design your own experiment (Task3.py) (contains Extension) - This script performs everything outlined in task 3 in project description. Everything is set up user can directy run the script. 

>> In order to change the number of epochs the user can change value in line 106 and to change the number of neural network combinations change value in line 129. 

>> The extension is that I have explored 5 different dimensions and 100 differnt neural network. The yml file containing the different combination of dimensions and their respective accuracies (after 5 epochs of training) is saved in the current directory. The neural networks are saved in current_dir/results/task3/.

- The results folder has the saved network in each task in format - /results/task1/,/results/task2/...
- For task 3 results the folder also contains the yml data for evaulation on 100 neural networks along with the best nerual network saved.
- Digit_Greekletter_custom_data.rar contains the self prepared test data for task 1 and task 2.





