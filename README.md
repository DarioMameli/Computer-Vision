# SportVideoAnalysis-SADVision
Sport Video Analysis to obtain semantic information on game, boundaries and context

YOLO training: _YOLO_train_yolov8_instance_segmentation.ipynb_
YOLO inference: _YOLO_inference.ipynb_
runs/segment/predict for the images with overlapped segments and binary masks produced by the inference of YOLO.

_map.txt_, _miou.txt_ in performance measurements for the latest evaluation of the system.

_main.cpp_ for the main detection and segmentation using also deep learning.
Results for final evaluation in _/Masks_, with the latest and best run.
_/ImagesDetection_ for the images displaying the detections of the latest and best run.

_mainDetectionOpenCV.cpp_ for the detection with any of the three functions defined using OpenCV.
Bounding boxes and images with detections for the OpenCV detection in runs.

STEPS to run:
1) choose either main.cpp or mainDetectionOpenCV.cpp as main file modifying accordingly the CMakeLists.txt file.
2) Pass all paths and parameters with program arguments as follows:
   * Input arguments for _main.cpp_: <br /> ./SportVideoAnalysis-SADVision.exe --dataset_path=\<value\> --groundTruthMasks=\<value\> --segmentedMasksPath_pretrained=\<value\>
   * Input arguments for _mainDetectionOpenCV.cpp_: <br /> ./SportVideoAnalysis-SADVision.exe --dataset_path=\<value\> --groundTruthMasks=\<value\> --choice=<0-2>
3) (if main.cpp) run _YOLO_inference.ipynb_ to generate new binary masks.
4) run main.


Output common folders are:
* _../Masks/_ that contains the bounding boxes file, the binary and color segmentation images
* _../ImagesDetection/_ that contains the images with the displayed bounding boxes (results of detection on images)
* _../Sport_scene_dataset/groundTruthDetectionImages/_ that contains ground truth images with displayed bounding boxes 
* _../performance_measurements_ that contains the performance of mAP and mIoU of the all dataset for the last run (overwritten each execution)

When you run the _mainDetectionOpenCV.cpp_ the following folders are also generated: 
* _../runs/"+algorithm[choice]+"/Images/_ that contains all bounding boxes txt files
* _../runs/"+algorithm[choice]+"/BBoxes/_ that contains all images with displayed bounding boxes

