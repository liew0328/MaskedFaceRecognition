# Read Me
1. Ensure all the library is installed, including  pytorh, opencv, facenet_pytorch, kivy, kivymd.
2. Run the application with cmd/terminal with "python main.py"
3. Enjoy!

## Funtions
* Menu
  - List of menu will be shown when the drawer menu at top left corner is clicked
* Verification screen
  - if the verify button is clicked, the result will be shown in the result label
  - Euclidean distance larger than 0.8 will be considered as "unknown"
* Registration screen
  - Insert a unique name and click the capture button would bring user to the capture screen
  - In the capture screen, image will be stored into the ~/FaceImages/userName folder if the capture button is clicked
* Training screen
  - All of the face in FaceImages would be train and calculate their encoding when train button is clicked
