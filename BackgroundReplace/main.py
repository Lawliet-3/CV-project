import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
#cap.set(cv2.CAP_PROP_FPS, 60)

segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

listImg = os.listdir('D:\\UIT\\Computer Vision\\opencv\\BackgroundReplace\\images')

imgList = []
for imgPath in listImg:
    img = cv2.imread(f'D:\\UIT\\Computer Vision\\opencv\\BackgroundReplace\\images\\{imgPath}')
    imgList.append(img)

def import_and_modify_bg():
    # Open file dialog to select an image
    img_path = filedialog.askopenfilename(title="Select a selfie image", filetypes=[("Image files", "*.jpg *.png")])

    if img_path:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 480))  # Resize to match camera resolution

        # Convert image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply background subtraction
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        mask = bg_subtractor.apply(gray_img)

        # Create a green screen background (pure green color)
        green_screen = np.zeros_like(img)
        green_screen[:, :] = (0, 255, 0)  # Set RGB value to green

        # Replace the background with the green screen background
        result = cv2.bitwise_and(green_screen, green_screen, mask=mask) + cv2.bitwise_and(img, img, mask=~mask)

        # Save the result as a PNG image
        output_path = img_path.replace('.jpg', '_greenscreen.png').replace('.png', '_greenscreen.png')
        cv2.imwrite(output_path, result)

        print(f"Image with green screen background saved: {output_path}")


# Function to start the camera loop
def start_camera():
    global camera_running
    camera_running = True
    root.destroy()

# Create the main window
root = tk.Tk()
root.title("Background Removal App")

# Create a label and combo box for background selection
background_label = ttk.Label(root, text="Select Background:", font=("Helvetica", 14))
background_label.pack(pady=10)

background_combo = ttk.Combobox(root, values=listImg, font=("Helvetica", 12))
background_combo.pack(pady=5)

# Create buttons for the two options
import_button = ttk.Button(root, text="Import Image and Modify", command=import_and_modify_bg)
import_button.pack(pady=10)

start_button = ttk.Button(root, text="Start Camera", command=start_camera)
start_button.pack(pady=5)

# Configure ttk styling options for better appearance
style = ttk.Style()
style.configure("TLabel", foreground="blue", background="white")
style.configure("TCombobox", foreground="black", background="white")
style.configure("TButton", foreground="black", background="blue", font=("Helvetica", 12))

# Start the GUI event loop
root.mainloop()

# Only proceed if the camera was started from the GUI
if camera_running:
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    segmentor = SelfiSegmentation()
    fpsReader = cvzone.FPS()

    IndexImg = 0

    while True:
        success, img = cap.read()
        imgOut = segmentor.removeBG(img, imgList[IndexImg], threshold=0.4)

        imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
        _, imgStacked = fpsReader.update(imgStacked, color=(0, 0, 255))

        cv2.imshow("Image", imgStacked)
        key = cv2.waitKey(1)

        if key == ord('a'):
            if IndexImg > 0:
                IndexImg -= 1
        elif key == ord('d'):
            if IndexImg < len(imgList) - 1:
                IndexImg += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
