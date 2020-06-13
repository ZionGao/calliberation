import cv2
import numpy as np
import glob
import laspy.file
import laspy.header
import math
import random
import warnings

warnings.filterwarnings('ignore')

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 获取标定板角点的位置
objp = np.zeros((8 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:144:8j, 0:144:8j].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
# print(objp)

obj_points = []  # 存储3D点
img_points = []  # 存储2D点

images = glob.glob("data/*.jpg")
i=0;
print(len(images))
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (8, 8), None)
    #print(corners)

    if ret:

        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        #print(corners2)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (8, 8), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
        i+=1;
        cv2.imwrite('conimg'+str(i)+'.jpg', img)
        cv2.waitKey(1500)

print(len(img_points))
cv2.destroyAllWindows()

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("mtx:\n", mtx) # 内参数矩阵
print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
print("tvecs:\n", tvecs ) # 平移向量  # 外参数

print("-----------------------------------------------------")


object_3d_points = np.array((
                             [2000, 22, -51],
                             [1986, 422, -58],
                             [2017, -382, -49],
                             [2959, 454, 252],
                             [2981, 52, 232],
                             [2985, -358, 242],
                             [3906, 445, 533],
                             [3968, -374, 517]
), dtype=np.double)
object_2d_point = np.array((
                            [824, 604],
                            [542, 601],
                            [1106, 603],
                            [627, 465],
                            [820, 467],
                            [1012, 467],
                            [685, 383],
                            [974, 387]
), dtype=np.double)
# object_3d_points = np.array((
#         # [5537,284,234],
#         [5502,520,218],
#         [13743,-618,1775],
#         [7109,-873,311],
#         [15503,3359,720],
#         [16049,3958,123],
#     # [6510,-1463,692],
#     [5377,259,-537],
#     [6921,-1061,463]
#
#     # [5345,518,-548]
#     # [22129,3010,3189]
#     ), dtype=np.double)
# object_2d_point = np.array((
#         # [1817,1050],
#         [1686,1044],
#         [2091,789],
#         [2303,1044],
#         [1332,1003],
#         [1275,1132],
    # [2600,861],
    # [1822,1467],
    # [2341,975]

    # [2638,1286]

    # [1579,720]
        # [1941, 1059],
        # [2139, 966],
        # [2280,389],
        # [1315,1007],
    # ), dtype=np.double)
camera_matrix = mtx
dist_coefs = dist
# 求解相机位姿
found, rvec, tvec = cv2.solvePnP(object_3d_points, object_2d_point, camera_matrix, dist_coefs)
rotM = cv2.Rodrigues(rvec)[0]
camera_postion = -np.matrix(rotM).T * np.matrix(tvec)
print(camera_postion.T)
# 验证根据博客http://www.cnblogs.com/singlex/p/pose_estimation_1.html提供方法求解相机位姿
# 计算相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。旋转顺序z,y,x
thetaZ = math.atan2(rotM[1, 0], rotM[0, 0])*180.0/math.pi
thetaY = math.atan2(-1.0*rotM[2, 0], math.sqrt(rotM[2, 1]**2 + rotM[2, 2]**2))*180.0/math.pi
thetaX = math.atan2(rotM[2, 1], rotM[2, 2])*180.0/math.pi
# 相机坐标系下值
x = tvec[0]
y = tvec[1]
z = tvec[2]
# 进行三次旋转
def RotateByZ(Cx, Cy, thetaZ):
    rz = thetaZ*math.pi/180.0
    outX = math.cos(rz)*Cx - math.sin(rz)*Cy
    outY = math.sin(rz)*Cx + math.cos(rz)*Cy
    return outX, outY
def RotateByY(Cx, Cz, thetaY):
    ry = thetaY*math.pi/180.0
    outZ = math.cos(ry)*Cz - math.sin(ry)*Cx
    outX = math.sin(ry)*Cz + math.cos(ry)*Cx
    return outX, outZ
def RotateByX(Cy, Cz, thetaX):
    rx = thetaX*math.pi/180.0
    outY = math.cos(rx)*Cy - math.sin(rx)*Cz
    outZ = math.sin(rx)*Cy + math.cos(rx)*Cz
    return outY, outZ
(x, y) = RotateByZ(x, y, -1.0*thetaZ)
(x, z) = RotateByY(x, z, -1.0*thetaY)
(y, z) = RotateByX(y, z, -1.0*thetaX)
Cx = x*-1
Cy = y*-1
Cz = z*-1
# 输出相机位置
print(Cx, Cy, Cz)
# 输出相机旋转角
print(thetaX, thetaY, thetaZ)
# 对第五个点进行验证
Out_matrix = np.concatenate((rotM, tvec), axis=1)


pixel = np.dot(camera_matrix, Out_matrix)

def get_pixel(p1,arr1):
    pi1 = np.dot(p1, np.array([ arr1[0] ,arr1[1], arr1[2],1], dtype=np.double))
    pi2 = pi1/pi1[2]
    return pi2


# pixel1 = np.dot(pixel, np.array([ -333 ,305, 2186,1], dtype=np.double))
# pixel2 = pixel1/pixel1[2]
# print(pixel2)

img = cv2.imread("IMG.jpg")
f = laspy.file.File("pointCloud/10.las", mode="r")

z = f.get_points()
dep = z.tolist()
dep = list(map(lambda a:a[0][0],z))
dep = [ one for one in dep if one != 0]
maxd = max(dep)
mind = min(dep)
print(mind)





def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        curpoint = np.random.choice(z, 10000)
        for pp in curpoint:
            arr = [pp[0][0],pp[0][1],pp[0][2]]
            colour = int((pp[0][0]-mind)/(maxd-mind)*255)
            x,y,ss = get_pixel(pixel,arr)
            x = int(x)
            y = int(y)
            xy = "%d,%d" % (x, y)
            if colour <100:
                colour==0
            cv2.circle(img, (x, y), 1, (int(colour/2), int(colour/4), int(colour/5)), thickness=1)
            # cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
            #         1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)

while (True):
    try:
        cv2.waitKey(100)
    except Exception:
        cv2.destroyWindow("image")
        break

cv2.waitKey(0)
cv2.destroyAllWindow()
