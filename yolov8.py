# Kieren Hussey
# Starter code came from https://www.youtube.com/watch?v=O9Jbdy5xOow&list=PLJA8jVJaGxqlscYUxiXm0mRBHKNFDPyJE&index=1

# Using YOLOv8 to detect people in a video feed and draw bounding boxes around them. Using the Pytorch model from ultralytics https://docs.ultralytics.com/tasks/detect/#models 



import torch # deep learning framework
import numpy as np # numerical computing
import cv2 # computer vision
from time import time # time the FPS
from ultralytics import YOLO # YOLOv8 model


from supervision.draw.color import ColorPalette, Color
from supervision import Detections, BoxAnnotator 


class ObjectDetection: 

  def __init__(self, capture_index): # Constructor for the ObjectDetection class and initializes the object

    self.capture_index = capture_index # Stores the camera index or video file path to be used for capturing frame

    # Checking if there is a gpu available and if its cuda(nvidia) compatible
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using Device: ", self.device)
    if torch.cuda.is_available():
      print("GPU Name:", torch.cuda.get_device_name(0))



    self.model = self.load_model() # Loads the YOLO model

    self.CLASS_NAMES_DICT = self.model.model.names # Stores the class names from the YOLO model
    
    # This draws a green box around detected object
    green_color = Color(0, 255, 0)
    self.box_annotator = BoxAnnotator(color=green_color, thickness=3, text_scale=1.5)


  # Load_model method created 
  def load_model(self): 

    model = YOLO("yolov8m.pt") # Loads the pytorch model
    model.fuse() # Fuses for better performance 

    return model 
  

  # Predict method created
  def predict(self, frame):

    results = self.model(frame) # Feeds the frame to the YOLO model

    return results
  

  # Bounded Boxes method created
  def plot_bboxes(self, results, frame):

    xyxys = []
    confidences = []
    class_ids = []

    #  Extract detections for person class
    for result in results[0]:
      class_id = result.boxes.cls.cpu().numpy().astype(int)

      if class_id == 0:

        xyxys.append(result.boxes.xyxy.cpu().numpy())
        confidences.append(result.boxes.conf.cpu().numpy())
        class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

    # Setup detections for visualization
    detections = Detections(
      xyxy=results[0].boxes.xyxy.cpu().numpy(),
      confidence=results[0].boxes.conf.cpu().numpy(),
      class_id=results[0].boxes.cls.cpu().numpy().astype(int),
      )
    
    print(detections)

    # Format custom labels
    self.labels = [f"{self.CLASS_NAMES_DICT[cid]} {conf:0.2f}" for cid, conf in zip(detections.class_id, detections.confidence)]


    # Annotate and display frame
    frame = self.box_annotator.annotate(frame, detections=detections, labels=self.labels)

    return frame
  


  # This method allows the object to be called as a function
  def __call__(self):

    cap = cv2.VideoCapture(self.capture_index) # Open video capture using OpenCV
    assert cap.isOpened()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    # Inside this while loop its getting the time in frames per second
    while True:

      start_time = time()

      ret, frame = cap.read()
      assert ret

      results = self.predict(frame)
      frame = self.plot_bboxes(results, frame)

      end_time = time()
      fps = 1/np.round(end_time - start_time, 2)

      cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)


      cv2.imshow('YOLOv8 Detection', frame)

      if cv2.waitKey(5) & 0xFF == 27:

        break

    cap.release()
    cv2.destroyAllWindows()

detector = ObjectDetection(capture_index=0)
detector()
  

  


  


