import cv2
import numpy as np
import glob
import laspy.file
import laspy.header
import math
import random
import warnings

warnings.filterwarnings('ignore')


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

def get_pixel(p1,arr1):
    pi1 = np.dot(p1, np.array([ arr1[0] ,arr1[1], arr1[2],1], dtype=np.double))
    pi2 = pi1/pi1[2]
    return pi2

def callibrate_camera(file):
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((8 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:144:8j, 0:144:8j].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    # print(objp)

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点

    images = glob.glob(file)
    # images = glob.glob("data/*.jpg")
    i = 0
    print("读入图像{}张".format(len(images)))
    # cv2.namedWindow("image")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (8, 8), None)
        # print(corners)

        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            # print(corners2)
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)

            cv2.drawChessboardCorners(img, (8, 8), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
            i += 1;
            cv2.imwrite('tmpImage/conimg' + str(i) + '.jpg', img)
    #         cv2.imshow("image", img)
    #         cv2.waitKey(1500)
    # cv2.destroyWindow("image")

    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    print("ret:", ret)
    print("mtx:\n", mtx)  # 内参数矩阵
    print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
    print("tvecs:\n", tvecs)  # 平移向量  # 外参数
    print("                                                       ")
    print("                      相机标定结束                       ")
    print("                                                       ")
    return mtx,dist


def callibrate_camera_and_lidar(mtx,dist):
    args = input("请输入相机大地坐标：（x,y,z） 单位mm\n")
    XYZ = args.split(",")
    camX,camY,CamZ = map(int, XYZ)
    camXYZ = [camX,camY,CamZ]

    flag = True
    count = 1
    points = np.empty(shape=[0,3],dtype=np.double)
    pixels = np.empty(shape=[0,2],dtype=np.double)
    while flag:

        a1 = input("请输入标定大地坐标及相应的像素坐标，按 Q 退出：\n")
        if a1 == "Q":
            break
        if a1 != "":
            text = a1.split("\t")
            tmpXYZ = text[0].split(",")
            tmpUV = text[-1].split(",")
            tmpX, tmpY, tmpZ = map(int, tmpXYZ)
            tmpPoint = np.array([tmpX, tmpY, tmpZ])
            tmpU, tmpV = map(int, tmpUV)
            tmpPixel = np.array([tmpU, tmpV])
            print("第{}点大地坐标：X - {},Y - {},Z - {}\n".format(count,tmpX, tmpY, tmpZ))
            print("第{}点雷达坐标：Lx - {},Ly - {},Lz - {}\n".format(count,tmpX-camX, tmpY-camY, tmpZ-CamZ))
            print("第{}点像素坐标：U - {},V - {}\n".format(count,tmpU, tmpV))
            points = np.row_stack((points, tmpPoint))
            pixels = np.row_stack((pixels, tmpPixel))
            count += 1

    # object_3d_points = np.array((
    #     # [5537,284,234],
    #     [5502, 520, 218],
    #     [13743, -618, 1775],
    #     [7109, -873, 311],
    #     [15503, 3359, 720],
    #     [16049, 3958, 123],
    #     # [6510,-1463,692],
    #     [5377, 259, -537],
    #     [6921, -1061, 463]
    #
    #     # [5345,518,-548]
    #     # [22129,3010,3189]
    # ), dtype=np.double)
    # object_2d_point = np.array((
    #     # [1817,1050],
    #     [1686, 1044],
    #     [2091, 789],
    #     [2303, 1044],
    #     [1332, 1003],
    #     [1275, 1132],
    #     # [2600,861],
    #     [1822, 1467],
    #     [2341, 975]
    #
    #     # [2638,1286]
    #
    #     # [1579,720]
    #     # [1941, 1059],
    #     # [2139, 966],
    #     # [2280,389],
    #     # [1315,1007],
    # ), dtype=np.double)

    object_3d_points = points
    object_2d_point = pixels
    dist_coefs = dist
    # 求解相机位姿
    found, rvec, tvec = cv2.solvePnP(object_3d_points, object_2d_point, mtx, dist_coefs)
    rotM = cv2.Rodrigues(rvec)[0]
    camera_postion = -np.matrix(rotM).T * np.matrix(tvec)
    # 计算相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。旋转顺序z,y,x
    thetaZ = math.atan2(rotM[1, 0], rotM[0, 0])*180.0/math.pi
    thetaY = math.atan2(-1.0*rotM[2, 0], math.sqrt(rotM[2, 1]**2 + rotM[2, 2]**2))*180.0/math.pi
    thetaX = math.atan2(rotM[2, 1], rotM[2, 2])*180.0/math.pi
    # 相机坐标系下值
    x = tvec[0]
    y = tvec[1]
    z = tvec[2]
    (x, y) = RotateByZ(x, y, -1.0 * thetaZ)
    (x, z) = RotateByY(x, z, -1.0 * thetaY)
    (y, z) = RotateByX(y, z, -1.0 * thetaX)
    Cx = x * -1
    Cy = y * -1
    Cz = z * -1
    # 输出相机位置
    print("相机相对位置 \nCx:{}\nCy:{}\nCz:{}\n".format(Cx, Cy, Cz))
    # 输出相机旋转角
    print("相机旋转角 \nthetaX:{}\nthetaY:{}\nthetaZ:{}\n".format(thetaX, thetaY, thetaZ))
    # 对第五个点进行验证
    return rotM,tvec,camXYZ

def point2pixel(rotM, tvec,camera_matrix,camXYZ):
    a3 = input("输入大地坐标：（x,y,z） 单位mm\n")
    Out_matrix = np.concatenate((rotM, tvec), axis=1)
    pixel = np.dot(camera_matrix, Out_matrix)

    tmpXYZ = a3.split(",")
    tmpX, tmpY, tmpZ = map(int, tmpXYZ)
    tmpPoint = np.array([tmpX, tmpY, tmpZ])
    tmpPoint = np.append(tmpPoint,[1.])
    # pixel1 = np.dot(pixel, np.array([-333, 305, 2186, 1], dtype=np.double))
    pixel1 = np.dot(pixel, tmpPoint)
    pixel2 = pixel1 / pixel1[2]

    print("大地坐标：X = {},Y = {},Z = {}\n".format(tmpX, tmpY, tmpZ))
    print("雷达坐标：Lx = {},Ly = {},Lz = {}\n".format( tmpX - camXYZ[0], tmpY - camXYZ[1], tmpZ - camXYZ[2]))
    print("映射像素坐标：U = {},V = {}\n".format( pixel2[0], pixel2[1]))

    a4 = input("请输入真实像素坐标：（u,v） \n")
    tmpUV = a4.split(",")
    tmpU, tmpV = map(int, tmpUV)

    bias = math.sqrt((pixel2[0] - tmpU) ** 2 + (pixel2[1] - tmpV) ** 2)
    print("该点映射像素偏差： {}\n".format( bias))

def pixel2point(rotM, tvec,camera_matrix):
    f1 = input("输入图片文件：\n")
    f2 = input("输入点云文件：\n")
    img = cv2.imread("1.jpg")
    f = laspy.file.File("pointCloud/11.las", mode="r")

    Out_matrix = np.concatenate((rotM, tvec), axis=1)
    pixel = np.dot(camera_matrix, Out_matrix)

    height = img.shape[0]
    width = img.shape[1]
    pixelNum = width*height

    points = f.get_points()
    pointNum = len(points)
    X = f.get_x()
    Y = f.get_y()
    Z = f.get_z()

    print("读入图像尺寸{}*{},像素点共{}个\n".format(width, height, pixelNum))
    print("读入点云数据共{}点，X轴范围：{} - {},Y轴范围：{} - {},Z轴范围：{} - {}\n".format(pointNum, max(X),min(X), max(Y),min(Y),max(Z),min(Z)))

    pointMap = np.zeros((width, height, 3),dtype=np.float)

    print("正在融合图像与点云...")
    for p in points:
        arr = [p[0][0],p[0][1],p[0][2]]
        x,y,ss = get_pixel(pixel,arr)
        x = int(x)
        y = int(y)

        if (y >= height or y <= 0 or x>=width or x <=0):
            continue

        pointMap[x, y, 0] = p[0][0]
        pointMap[x, y, 1] = p[0][1]
        pointMap[x, y, 2] = p[0][2]


    while True:

        f3 = input("输入像素坐标：\n")
        if f3 == "Q":
            break
        tmpUV = f3.split(",")
        tmpU, tmpV = map(int, tmpUV)
        print("该像素点映射坐标 X={}, Y={}, Z={}\n".format(pointMap[tmpU, tmpV, 0], pointMap[tmpU, tmpV, 1], pointMap[tmpU, tmpV,  2]))

        a4 = input("请输入真实像素坐标：（X,Y,Z） \n")
        tmpXYZ = a4.split(",")
        tmpX, tmpY, tmpZ  = map(int, tmpXYZ)

        bias = math.sqrt((tmpX - pointMap[tmpU, tmpV, 0]) ** 2 + (tmpY - pointMap[tmpU, tmpV, 1]) ** 2 + (tmpZ - pointMap[tmpU, tmpV, 2]) ** 2)
        print("该点映射坐标偏差： {}\n".format(bias))















def perform():
    print("===" * 18)
    print("***" * 18)
    print("===" * 18)
    flag = True
    global mtx1,dist1,rotM1,tvec1
    while flag:
        print("===" * 18)
        print("***" * 18)
        print("===" * 18)
        a1 = input("输入演示功能选项：\n1）相机内参标定 2）相机雷达联合标定 3）3D->2D 4) 2D->3D\n按 Q 退出\n")
        print("===" * 18)
        print("***" * 18)
        print("===" * 18)
        if a1 == "1":
            a2 = input("输入标定图像路径：\n")
            mtx1,dist1 = callibrate_camera(a2)
        if a1 == "2":
            rotM1,tvec1,camXYZ = callibrate_camera_and_lidar(mtx1,dist1)
        if a1 == "3":
            point2pixel(rotM1,tvec1,mtx1,camXYZ)
        if a1 == "4":
            pixel2point(rotM1,tvec1,mtx1)
        if a1 == "Q":
            flag = False






if __name__ == '__main__':
    perform()



