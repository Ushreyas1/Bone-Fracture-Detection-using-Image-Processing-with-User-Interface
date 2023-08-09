import cv2
import numpy as np
from pre_process import _reshape_img, get_model


img_name = "F81"

"""
Currently, `img_name` will be used to get a resized image from the `images/resized` folder
and the original image from the `images/Fractured Bone` folder, so it expects the same named image file
to be available in both folders.

All names starting with F{n} are available in both folders. 1 <= n <= 100
"""

model_name = "ridge_model"

img_file = 'images/resized/{}'.format(img_name)
orig_img = 'images/Fractured Bone/{}'.format(img_name)

# for image read
try:
    img_t = cv2.imread(img_file + ".jpg", cv2.IMREAD_COLOR)
    img = cv2.imread(orig_img + ".jpg", cv2.IMREAD_COLOR)
    shape = img.shape
except (AttributeError, FileNotFoundError):
    try:
        img_t = cv2.imread(img_file + ".JPG", cv2.IMREAD_COLOR)
        img = cv2.imread(orig_img + ".JPG", cv2.IMREAD_COLOR)
        shape = img.shape
    except (AttributeError, FileNotFoundError):
        img_t = cv2.imread(img_file + ".png", cv2.IMREAD_COLOR)
        img = cv2.imread(orig_img + ".png", cv2.IMREAD_COLOR)
        shape = img.shape

# details of Image
print("\nShape: ", shape)
print("\nSize: ", img.size)
print("\nDType: ", img.dtype)

# ==============Manual edge detect=====================
def segment_img(_img, limit):
    for i in range(_img.shape[0] - 1):
        for j in range(_img.shape[1] - 1):
            if int(_img[i, j + 1]) - int(_img[i, j]) >= limit:
                _img[i, j] = 0
            elif int(_img[i, j - 1]) - int(_img[i, j]) >= limit:
                _img[i, j] = 0

    return _img


# ======================================================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# gray=segment_img(gray,15)
cv2.imshow("GrayEdited", gray)
median = cv2.medianBlur(gray, 5)

model = get_model(model_name)
pred_thresh = model.predict([_reshape_img(img_t)])
print(pred_thresh[0])
_, threshold_img = cv2.threshold(median, pred_thresh[0], 255, cv2.THRESH_BINARY)
cv2.imshow("threshold", threshold_img)

initial = []
final = []
line = []

for i in range(0, gray.shape[0]):
    tmp_initial = []
    tmp_final = []
    for j in range(0, gray.shape[1] - 1):
        if threshold_img[i, j] == 0 and threshold_img[i, j + 1] == 255:
            tmp_initial.append((i, j))
        if threshold_img[i, j] == 255 and threshold_img[i, j + 1] == 0:
            tmp_final.append((i, j))

    x = [each for each in zip(tmp_initial, tmp_final)]
    x.sort(key=lambda each: each[1][1] - each[0][1])
    try:
        line.append(x[len(x) - 1])
    except IndexError:
        pass

err = 15
danger_points = []
dist_list = []

for i in range(1, len(line) - 1):
    dist_list.append(line[i][1][1] - line[i][0][1])
    try:
        prev_ = line[i - 3]
        next_ = line[i + 3]
        dist_prev = prev_[1][1] - prev_[0][1]
        dist_next = next_[1][1] - next_[0][1]
        diff = abs(dist_next - dist_prev)
        if diff > err:
            data = (diff, line[i])
            if len(danger_points):
                prev_data = danger_points[-1]
                if abs(prev_data[0] - data[0]) > 2 or data[1][0] - prev_data[1][0] != 1:
                    danger_points.append(data)
            else:
                danger_points.append(data)
    except Exception as e:
        pass

for i in range(0, len(danger_points) - 1, 2):
    try:
        start_rect = danger_points[i][1][0][::-1]
        start_rect = (start_rect[0] - 40, start_rect[1] - 40)
        end_rect = danger_points[i + 1][1][1][::-1]
        end_rect = (end_rect[0] + 40, end_rect[1] + 40)
        cv2.rectangle(img, start_rect, end_rect, (0, 255, 0), 2)
    except:
        print("Pair not found")

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1)
fig2, ax3 = plt.subplots(1, 1)

x = np.arange(1, gray.shape[0] - 1)
y = dist_list

cv2.calcHist(gray,[0],None,[256],[0,256])

try:
	ax1.plot(x,y)
except:
    print("Could not plot")
    
img = np.rot90(img)
ax2.imshow(img)

ax3.hist(gray.ravel(), 256, [0, 256])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
