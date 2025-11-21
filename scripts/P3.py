import cv2 
import numpy as np
import matplotlib.pyplot as plt


      
def findingContours(frame):
       
   x,y,w,h=0,0,0,0
   contours,_=cv2.findContours(frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
   for contour in contours:
      area= cv2.contourArea(contour)

      if area>500:

         #cv2.drawContours(res,contour,-1,(255,0,0),3)
         peri=cv2.arcLength(contour,True)
         approx=cv2.approxPolyDP(contour,0.02*peri,True)
         x,y,w,h= cv2.boundingRect(approx)
         
   # Returning mid point of the blob
   return x+w/2,y+h/2,w,h
      

def Masking( frame):

   #Set threshold to filter only orảnge color
   lower= np.array([0,225,200])
   upper= np.array([20,255,255])
   mask= cv2.inRange( frame, lower, upper)
   
   # Dilate
   k= np.ones((10, 10))
   maskDilated = cv2.dilate(mask, k)
   maskC= cv2.bitwise_and(frame,frame,mask=maskDilated)
   maskC= cv2.resize(maskC,(800,800))
   
   return maskDilated,maskC

#Performs required image processing to get ball coordinated 
def drawing(frame,points):
   
   for i in range (1,len(points)):
      cv2.line(frame, points[i - 1], points[i], (0, 0, 255), 2)

# open video
cap = cv2.VideoCapture("ball_trimmed.mp4")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # or 'XVID' for .avi
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))


points = []
predicted_points = []
predicted = np.zeros((2, 1), np.float32)
count = 0

# assuming you already have these helper functions defined:
# Masking(), findingContours(), Estimate(), drawing()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask, _ = Masking(hsv)
    _, maskC = Masking(hsv)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        count += 1
        cv2.imwrite(f'ball{count}.jpg', maskC)

    nextPoint = []
    x, y, w, h = findingContours(mask)
    nextPoint.append((int(x), int(y)))

    for i in nextPoint:
        points.append(i)

    # avoiding the drawing of lines when ball disappears from the screen
    if (0, 0) in points:
        points.remove((0, 0))

    # limit visible trail length
    '''
    if len(points) > 15:
        del points[0]
    '''
    drawing(frame, points)

    # Draw actual coords

    cv2.circle(frame, (int(x), int(y)), 20, [0, 0, 255], 2, 7)
    #cv2.line(frame, (int(x), int(y + 20)), (int(x + 50), int(y)), [0, 0, 0], 2, 7)
    cv2.putText(frame, "Actual", (int(x + 50), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255])

    out.write(frame)
    cv2.imshow('final', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---- Plot trajectory after video ----
if len(points) > 0 or len(predicted_points) > 0:
    plt.figure(figsize=(8, 5))
    if len(points) > 0:
        x_actual, y_actual = zip(*points)
        plt.plot(x_actual, y_actual, 'ro', 
         markersize=10,        # controls the overall circle size
         markerfacecolor='none',  # hollow (no fill)
         markeredgewidth=2,     # thickness of the ring edge
         label='Actual')

    plt.gca().invert_yaxis()  # match image coordinates
    plt.title("Ball Trajectory")
    plt.xlabel("X position (pixel)")
    plt.ylabel("Y position (pixel)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("trajectory.png", dpi=200)
    plt.show()

    print("✅ Trajectory graph saved as 'trajectory.png'")
else:
    print("⚠️ No points recorded.")
   


   

      
 




