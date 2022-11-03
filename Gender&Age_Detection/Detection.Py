import cv2
import math
import argparse

def highlightFace(net, frame, conf_threshold=0.7): #The model, frame and confidence is passed as parameter(threshold for confidence is 70% in this case.)
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0] #Storing Height of frame
    frameWidth=frameOpencvDnn.shape[1]#Storing width of frame

    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300,300), [104,117,123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

parser = argparse.ArgumentParser()
parser.add_argument('--image')
args=parser.parse_args()

#Initialize protocol buffer and model
#To Detect Face
faceProto='opencv_face_detector.pbtxt'
faceModel='opencv_face_detector_uint8.pb'
#To Detect Age
ageProto="age_deploy.prototxt"
ageModel='age_net.caffemodel'
#To Detect Gender
genderProto='gender_deploy.prototxt'
genderModel='gender_net.caffemodel'

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)', '(60-100)']
genderList=['Male','Female']

#readNet() is used to load networks. 1st parameter contains models while 2nd carries network configuration
faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

video=cv2.VideoCapture(args.image if args.image else 0) #Detects age & gender if image is passed, or detects it from video
padding=20
while cv2.waitKey(1)<0:
    hasFrame,frame=video.read() #video.read() returns if the frame is correctly read, if it isn't correctly read, it comes out of the loop
    if not hasFrame:
        cv2.waitKey()
        break    

    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
        min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding):
        min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB = False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender}, {age},', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detectnig age and gender", resultImg)