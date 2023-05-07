Name - Satvik Tyagi

Operating system - Windows 10
IDE - Visual Studio Code

Time Travel days used - 0


Instructions:
1. I was getting the video frames using my phone camera using Iriun Webcam app. In order to do the same download and install the app
   on both your PC and phone and make sure both are connected to the same network. Finally ensure that in file main_proj3.cpp,
   cv::VideoCapture is set to 1 instead of 0.

2. Run the main_proj3.cpp file. Place one object in the camera frame.
	- Choice of key press 'x', 't', 'i'

		- If 'x' is pressed the program computes the following: >threholded image of object in frame.
									>cleanup of the thresholded image using dilation and erosion.
									>segmentation of the image with differnt object in different colors.
									>bounding box and axis of least central moment around the biggest 									 	contour.

		- If 't' is pressed the program goes into training mode and all of the computation same as when x is pressed. After that the 			user is prompted to enter the object name. Once the user types in name and presses enter a feature vector of the object in 			frame is saved in file "database.csv".

		- If 'i' is pressed the program goes into inference mode and all the computation is same as when x is pressed. After that the 			feature vector is calculated. Then the user is prompted to choose between nearest neighbor classification or K nearest 				neighbor classification where K=3. If the user presses 1 nearest neighbor classification is done and when the user presses 0 			KNN is performed. After this an output window opens up which has the object with its class name, bounding box and axis of  			least central moment.
		

LINK TO DEMO VIDEO: https://drive.google.com/drive/folders/1SUHa00btIEBzSO7NxWvXIb2gn_7Jjnu3?usp=sharing