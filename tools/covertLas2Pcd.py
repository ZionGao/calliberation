import numpy as np
import pcl.pcl_visualization
import pcl
import random,math

import laspy.file
import laspy.header




def getCloud():

    f = laspy.file.File("pointCloud/11.las",mode="r")


    z = f.get_points()

    point_count = len(z)
    cloud = pcl.PointCloud_PointXYZRGB()
    points = np.zeros((point_count, 4), dtype=np.float32)
    print(point_count)
    # Generate the data
    i = 0


    for point in z:
        if point[0][0] == 0:
            print(point[0])
            continue
        print(point[0])
        points[i][0] = point[0][0]
        points[i][1] = point[0][1]
        points[i][2] = point[0][2]
        r1 = point[0][0]
        g1 = point[0][0]
        b1 = point[0][0]

        r2 = math.ceil((float(r1)/65536)*256.0)
        g2 = math.ceil((float(g1)/65536)*256.0)
        b2 = math.ceil((float(b1)/65536)*256.0)
        rgb = (int(r2)) << 16 | (int(g2)) << 8 | (int(b2))

        points[i][3] = rgb
        i+=1
    out = points[0:i-1,:]

    cloud.from_array(out)
    return cloud

def main():

    cloud=getCloud()
    pcl.save(cloud,"pointCloud/11.pcd")


if __name__ == '__main__':
    main()