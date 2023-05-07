Name - Satvik Tyagi

Operating system - Windows 10
IDE - Visual Studio Code

Time Travel days used - 2

Instructions:
1. Run the "main_vid.cpp" file.

2. The user is shown a prompt: "Save image or not?: 1 or 0" > The user must type either '1' for Yes or '0' for No and press enter.
   These images are essentially frames where the chessboard corners are detected. 

3. The user is shown a prompt: "Augment Yes or No?: 1 or 0" > The user must type either '1' for Yes or '0' for No and press enter.

4. If the user selects '1' in the previous step, another prompt is shown: "Choose Augmentation: 'axes', 'cube', 'pillars':" the user must         type 'axes', 'cube' or 'pillars' (extension).

5. Once the user makes the choice the video starts and now there is 4 options for the user which are as follows:

	> User press 's': With chessboard in the frame when s is pressed the program extracts the chessboard corners. This process is repeated 			  5 times at least before the user is prompted: 
			  "Run Calibration on images: 1 or 0:" > The user must type either '1' for Yes or '0' for No and press enter.
			  If the user presses '1' the camera calibration is run on the images
			  If the user presses '0' the user can add more images. 
			  These same images on which calibration is run are saved if the user choses '1' in the first prompt. 

	> User presses 't': With chessboard in frame when 't' is pressed the program shows the image of the chessboard with the augmented 			    object
	
	> User presses 'h': With pattern in frame when 'h' is pressed the program shows the harris corner extracted image of the pattern 

	> User presses 'q': The program quits

LINK TO DEMO VIDEO: https://drive.google.com/file/d/1SPTSnWoLnqUXn0HUhQehw6XocJ_OOWGh/view?usp=share_link