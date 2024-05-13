# Knight Problem Program

This Python program allows you to find the set of all minimum-length sequences to move a knight piece from an initial cell to a final cell on a chessboard. The breadth-first search algorithm is employed to determine these minimum-length sequences.

## How to Use

Change to the current directory (where your Dockerfile is located)
```bash
cd /path/to/your/directory
```

Build the Docker image
```bash
docker build -t knight_problem .
```

Run the Docker container 
```bash
docker run -it knight_problem
```

## Input Parameters

**Input Parameters:** The program will prompt you to enter the following information:

1. **Size of the chessboard (default is 8):**
   - This parameter indicates the dimensions of the chessboard, representing the number of rows and columns. The default value is set to 8, which is the standard size of a chessboard.

2. **x-coordinate of the knight's starting position (default is 2):**
   - This parameter represents the horizontal position (column) on the chessboard where the knight will start its journey. The default value is set to 2.

3. **y-coordinate of the knight's starting position (default is 2):**
   - This parameter represents the vertical position (row) on the chessboard where the knight will start its journey. The default value is set to 2.

4. **x-coordinate of the target position (default is 4):**
   - This parameter represents the horizontal position (column) on the chessboard where the knight aims to reach. The default value is set to 4.

5. **y-coordinate of the target position (default is 3):**
   - This parameter represents the vertical position (row) on the chessboard where the knight aims to reach. The default value is set to 3.

When you run the program, it will prompt you to provide values for these parameters. If you do not input any values, the program will use the default values mentioned above. Adjusting these parameters allows you to simulate different scenarios on the chessboard, changing the starting and target positions as well as the size of the chessboard.


## Output

The program will display the set of all minimum-length sequences along with their lengths. Additionally, it will generate a Graphviz/DOT file named 'chessboard.jpg' illustrating the chessboard with highlighted shortest paths. You can check in the docker in the app you will see the image with the name chessboard.jpg.








# Tabular Data Classification

## Generating Dataset
Generate a dataset from simulated knight games with the below features. Some of the key features for valid knight moves, extracted from the above script, include:

- **Starting Position:** The x and y coordinates of the knight's starting position on the chessboard. Useful for encoding move validity based on origin.
- **Ending Position:** The x and y coordinates of the ending position after attempting a knight move. Critical for classification.
- **Board State:** A 2D matrix (8x8) indicating positions of all pieces on the board. Can simplify by just indicating the presence/absence of pieces in each cell.
- **Optimal Length:** Shortest possible path length precomputed using the above script using the BFS algorithm.
- **Move Path:** The L-shaped path of cells traversed from starting to ending position (e.g., d4-f5-f6).
- **Board Edge Proximity:** Binary indicators for starting/ending cells being close to or on edges.
- **Validity:** Target variable indicating if the current attempted move is valid or invalid based on chess rules (Binary classifier). This can be calculated by determining the distance between the starting & ending position since the knight moves in an L shape.

### Possible knight moves:
- (Board state 1, e3): Valid
- (Board state 1, g4): Valid
- (Board state 1, a1): Invalid (occupied by another piece)

## Dataset Collection
The dataset can be obtained from popular game sites like chess.com or Lichess.org by another approach:

### Alternative Method:
1. Download PGN (Portable Game Notation) files of chess games from websites like Lichess. This can be done through their API or by web scraping game data.
2. Parse the PGN files to extract all chess moves, looking for move numbers and chess notations like "Nf3," "Nc6," etc.
3. Filter out only the moves from knights, identified by any move starting with "N." This will yield a list of all knight moves.
4. Clean this list to remove duplicate moves, resulting in a list of unique valid knight moves.
5. Structure this list into a dataset with columns such as Move number, Full move (e.g., "1. Nf3"), Start position (e.g., "f1"), End position (e.g., "f3"), and Game ID.
6. Export this structured data to a CSV file, adding rows for each move.




Now that we have a dataset created/collected having the above features along with a target variable for move validity, the next steps involve augmenting data for invalid moves. Additionally, I aim to increase the dataset through synthetic dataset generation, aiming for a balanced class distribution. This will help prevent bias towards overrepresented classes by employing techniques like stratification, undersampling, and oversampling if needed.

### Preprocessing:
   * Coordinate Normalization
   * Handling Missing values
   * Binary Encoding/Label Encoding/One hot Vector


### Visualization
Analyze the dataset with the following objectives:
  * Explore common patterns in knight moves.
  * Examine the frequency of specific moves.
  * Investigate the correlation between moves and game outcomes.
  * Identify common trajectories.
  * Explore relationships between features and the validity of moves.

### Model Training
Train classifiers like SVM, logistic regression, XGBoost, LightGBM, and Random Forest to predict if a given move is valid or not.

## Metrics 
Binary classification models are evaluated using various metrics to assess their performance. Here are common metrics used for binary classification:
  * Accuracy
  * F1 Score
  * Precision
  * Recall
  * Confusion Metrics
  * AUC-ROC Curve

### Error Analysis
  * Examine instances where the model misclassifies moves.
  * Identify common patterns or challenges leading to misclassification.
  * Investigate specific scenarios where the model struggles.


## Cloud Services Deployment for ML Model

### Secure Deployment on AWS EC2 Instance using FastAPI, Docker, and HTTPS

1. **Model Preparation:**
   - Prepare and save the ML model in a serializable format (e.g., .pkl, .pt etc).
   
2. **FastAPI/Flask Application:**
   - Create a FastAPI or Flask application (`model.py`) with API endpoints for predictions(Asynchronous and Synchronous).
   - Define endpoints for both GET or POST requests to handle user input.

3. **Dockerization:**
   - Create a Dockerfile to encapsulate the application for deployment on any machine.
   - Maintain a `requirements.txt` file containing all dependencies.

4. **Secure Deployment on AWS EC2:**
   - Generate a key pair (`key.pem`) for secure access to the EC2 instance.
   - Connect to the instance via SSH using AWS CLI.
   - Install Docker on the instance, build, and run the Docker container.

5. **HTTPS Setup:**
   - Acquire a domain name and configure it to point to your EC2 instance's IP.
   - Use services like Let's Encrypt to obtain an SSL certificate.
   - Configure your FastAPI app to use HTTPS.

6. **S3 Bucket for Dataset Storage:**
   - Store the dataset in an S3 bucket using the `boto3` library.
   - Utilize the storage capabilities of the S3 bucket for efficient data management.

7. **Model Testing:**
   - Create test cases for model predictions before deployment.
   - Test the model at different levels, including unit, integration, system, and acceptance levels.
   - Perform testing at pre-deployment, deployment, and post-deployment stages.

8. **Alternative Deployment Options:**
   - Consider deploying the model on Amazon SageMaker for its managed, scalable, and user-friendly features.
   - Comparing the advantages of Amazon SageMaker with EC2 instances, noting the trade-offs between control and complexity.





## Other ideas 

### Object Detection:
Train a convolutional neural network model like Faster R-CNN or YOLO v3 to detect the knight piece in images of a chess board. Use a dataset of chess board images with bounding boxes annotated around the knight piece to train the model. Leverage transfer learning from a pre-trained model like ResNet-50. Good accuracy can be achieved. 
Set up a camera over a physical/Online chess board connected to a system running the detection model. As a game is played, run each video frame through the detection model to get (x,y) coordinates of the knight in each frame. By comparing coordinates across frames, trace out the path followed by the knight.
Compare the detected path to the shortest calculated path from the initial program. Detect discrepancies to identify if the player made a sub-optimal move.



### Object Tracking:
Employ visual object trackers like SORT (Simple Online and Realtime Tracking) to track the knight piece across video frames. Combine the relative motion vectors between frames with a chessboard scanner to have both absolute and relative representations of positions.
Use the frame-by-frame positions to verify if the move sequence sticks to the shortest path calculated by the initial program between start and end.


### Reinforcement Learning:
Define a Markov Decision Process for knight moves on a chess board. Represent board state as input, a legal knight moves as actions, number of moves as cost.
Train a policy gradient model like PPO to optimize the policy for choosing actions that minimize total moves between any start and end states.
The learned policy would match the shortest path found through search algorithms while generalizing across-board layouts.
