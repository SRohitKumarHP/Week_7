# Week_7
Task: Build a "normal vs anomalous" pipeline using an autoencoder's reconstruction. Segmentation, Measurement &amp; Anomaly Detection.

To perform these tasks, we require a platform to label the dataset or to use an available COCO model dataset.

Link to Ultralytics Platform for training and Labeling:
  https://platform.ultralytics.com/home

On this platform, you can annotate and use the labeled dataset to train any YOLO model.
Steps:
  1. Upload the dataset images by creating a New Dataset option.
  2. Choose the Annotate option in the left taskbar. Fill the required fields like dataset name, task type, and visibility public or private. Upload all your images, then click Create Dataset.

     
<img width="1515" height="708" alt="image" src="https://github.com/user-attachments/assets/3cb85727-d072-42b9-84d6-6e126298aa60" />

  3. After uploading all the images, you can start annotating them one by one until all the images get annotated.

  
<img width="1525" height="682" alt="image" src="https://github.com/user-attachments/assets/17651e6b-208c-48b4-802b-9642fe7735be" />


  4. Next, for going to the training part, click on the Train option '+' symbol and give a name to the training project.


<img width="1253" height="282" alt="image" src="https://github.com/user-attachments/assets/1b58a974-c7a6-4201-b26e-1f70a32770d3" />


  5. Open the training project and click on New Model.


<img width="987" height="612" alt="image" src="https://github.com/user-attachments/assets/9b6cec2e-6279-488c-8de4-f9b7379857e4" />


  6. Select the base Model. Choose the dataset from the dataset dropdown button in the New Model and click on Start Training.
  7. After completing the training, download the Model.

     
<img width="1511" height="690" alt="image" src="https://github.com/user-attachments/assets/6924a97e-b1a1-466f-b985-7a244ac62836" />


  7.1. After downloading the .pt model, rename the .pt file to 'metal_defect.pt'.
  7.2. Download the files from this GitHub to the place where you save the 'metal_defect.pt' file.
  8. Download Anaconda onto your PC. Download the suitable one that supports your operating system.
  9. Steps in Anaconda Prompt Window:
    1. Search for Anaconda Prompt on your PC
    2. In the Anaconda Prompt, create an environment by running the command:
     > create --name yolo_env1 python=3.12
    3. Then activate the environment and follow the commands:
     > conda activate yolo_env1
    4. In the Anaconda prompt window, set the path where the my_model folder is available.
    5. After setting the correct path, install the library
     > pip install ultralytics
    6. Use this link https://pytorch.org/get-started/locally/ for video nvidia GPU. Run this command:
     > pip3 install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu132
    7. Then run the command:
     > python yolo_segment.py --model=metal_defect.pt --source=<file_name>
    8. You can keep any <file_name> as per the provided above images and video file.
     To run the live video for detection run the command as:
     > python yolo_segment.py --model=metal_defect.pt --source=usb0
 
