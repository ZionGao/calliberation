#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
boost::mutex cloud_mutex;
using namespace pcl;
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;


//定义回调参数结构体
struct callback_args{
    // structure used to pass arguments to the callback function
    PointCloudT::Ptr clicked_points_3d;
//    pcl::PCLPointCloud2Ptr clicked_points_3d;
    pcl::visualization::PCLVisualizer::Ptr viewerPtr;
};
void pp_callback(const pcl::visualization::PointPickingEvent& event, void *args)
{
    struct callback_args * data = (struct callback_args *)args;//点云数据 & 可视化窗口
    if (event.getPointIndex() == -1)
        return;
    PointT current_point;
//    PointT current_point;
    event.getPoint(current_point.x, current_point.y, current_point.z);
    data->clicked_points_3d->points.push_back(current_point);
    //Draw clicked points in red:
    pcl::visualization::PointCloudColorHandlerCustom<PointT> red(data->clicked_points_3d, 255, 0, 0);
    data->viewerPtr->removePointCloud("clicked_points");
    data->viewerPtr->addPointCloud(data->clicked_points_3d, red, "clicked_points");
    data->viewerPtr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "clicked_points");
    std::cout << current_point.x << "," << current_point.y << "," << current_point.z << std::endl;

}
int main()
{
    std::string filename("/Users/gao/PycharmProjects/callibrate_camera/pointCloud/10.pcd");
    //visualizer
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("viewer"));

    if (pcl::io::loadPCDFile(filename, *cloud))
    {
        std::cerr << "ERROR: Cannot open file " << filename << "! Aborting..." << std::endl;
        return 0;
    }
    std::cout << cloud->points.size() << std::endl;

    cloud_mutex.lock();    // for not overwriting the point cloud

    // Display pointcloud:
    viewer->addPointCloud(cloud, "bunny_source");
    viewer->setCameraPosition(4000, 1000, 1000, 0, 0, 0, 0);

    // Add point picking callback to viewer:
    struct callback_args cb_args;
    PointCloudT::Ptr clicked_points_3d(new PointCloudT);
    cb_args.clicked_points_3d = clicked_points_3d;
    cb_args.viewerPtr = pcl::visualization::PCLVisualizer::Ptr(viewer);

    viewer->registerPointPickingCallback(pp_callback, (void*)&cb_args);


    std::cout << "Shift+click on three floor points, then press 'Q'..." << std::endl;

    // Spin until 'Q' is pressed:
    viewer->spin();
    std::cout << "done." << std::endl;

    cloud_mutex.unlock();

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}

