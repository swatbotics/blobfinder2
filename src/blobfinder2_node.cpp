// A node for 
//
// config/arguments:
//   - input image (eg /camera/rgb/image_raw)
//   - dilation/erosion amount (default 1)
//   - ROI (default whole window, otherwise x,y,w,h)
//   - minimum blob size (default 1)
//   - # blobs
//
// outputs
//   - /blobfinder/color_foo/blobs
//     - centroid, area, and principal component of each detected blob for color_foo
//   - /blobfinder/color_foo/image
//     - thresholded (and opened) image for each color


#include "ros/ros.h"
#include <blobfinder/MultiBlobInfo.h>
#include <blobfinder/MultiBlobInfo3D.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include "ColorLUT.h"
#include <iostream>
#include <assert.h>
#include <cmath>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

using sensor_msgs::Image;
using sensor_msgs::image_encodings::RGB8;
using sensor_msgs::image_encodings::MONO8;

using sensor_msgs::PointCloud2;
using sensor_msgs::PointField;

//////////////////////////////////////////////////////////////////////
// We will define a class later to handle each color separately.

class ColorHandler;

//////////////////////////////////////////////////////////////////////
// Helper for point cloud messages

class PointCloudHelper {
public:

#define CHECK_WARN(expected, actual)					\
  if ((expected) != (actual)) {						\
    ROS_WARN_STREAM_ONCE("Parsing PointCloud2 message: expected " <<	\
			 expected << " for " << #actual <<		\
			 " but got " << actual << ". " <<		\
			 "Message will be ignored");			\
    return false;							\
  }

  const PointCloud2* const msg;
  const bool ok;

  static bool valid_field(const PointField& field,
			  const std::string& desired_name,
			  int desired_offset,
			  int desired_datatype,
			  int desired_count) {

    CHECK_WARN(desired_name, field.name);
    CHECK_WARN(desired_offset, field.offset);
    CHECK_WARN(desired_datatype, field.datatype);
    CHECK_WARN(desired_count, field.count);
    return true;

  }

  static bool validate(const PointCloud2* msg) {

    if (!msg) { return false; }

    CHECK_WARN(4, msg->fields.size());
    
    return ( valid_field(msg->fields[0],   "x",  0, PointField::FLOAT32, 1) &&
	     valid_field(msg->fields[1],   "y",  4, PointField::FLOAT32, 1) && 
	     valid_field(msg->fields[2],   "z",  8, PointField::FLOAT32, 1) &&
	     valid_field(msg->fields[3], "rgb", 16, PointField::FLOAT32, 1) );

  }

  PointCloudHelper(const PointCloud2* m=0): 
    msg(m), ok(validate(msg)) 
  {}

  size_t sub2ind(size_t x, size_t y) const {
    if (!msg || !ok) { return 0; }
    return y * msg->row_step + x * msg->point_step;
  }

  const float* xyz(size_t x, size_t y) const {
    if (!msg) { return 0; }
    const unsigned char* buf = &(msg->data[sub2ind(x,y)]);
    return (const float*)buf;
  }

  const unsigned char* rgb(size_t x, size_t y) const {
    if (!msg) { return 0; }
    const unsigned char* buf = &(msg->data[sub2ind(x,y)+16]);
    return buf;
  }

};

//////////////////////////////////////////////////////////////////////
// Class for sending color blob messages

class BlobFinder {
public:

  // node handle for this node
  ros::NodeHandle nh;

  // lookup table being used
  ColorLUT lut;   

  // structuring element for morphological opening 
  cv::Mat strel; 

  // is this the first image?
  bool first_image; 

  // sub-rectangle used for image
  cv::Rect roi;

  // max blob count
  int max_blob_count;

  // min blob area
  int min_blob_area;

  // search area for 3D stuff
  int point_search_radius;

  // subscription for images
  ros::Subscriber image_sub;

  // subscription for points
  ros::Subscriber points_sub;

  // handler for each color
  std::vector<ColorHandler*> handlers;

  // constructor
  BlobFinder();

  // image callback
  void image_callback(const Image::ConstPtr& msg);

  // point cloud callback
  void points_callback(const PointCloud2::ConstPtr& msg);

  void setROI(int w, int h);
  void process_image(const cv::Mat& image_rgb, 
		     const PointCloudHelper& pch=PointCloudHelper());

};

//////////////////////////////////////////////////////////////////////
// Class for handling one color of a BlobFinder

class ColorHandler {
public:
  
  BlobFinder* parent;

  size_t cidx;
  
  bool active;

  size_t num_blobs_subscribers;
  size_t num_image_subscribers;
  size_t num_blobs3d_subscribers;
  
  ros::Publisher blobs_pub;
  ros::Publisher image_pub;
  ros::Publisher blobs3d_pub;

  ColorHandler(BlobFinder* b, size_t cidx);

  void subs_inc(size_t&, const ros::SingleSubscriberPublisher& ssp);
  void subs_dec(size_t&, const ros::SingleSubscriberPublisher& ssp);

  void get_pos3d(blobfinder::BlobInfo3D& b3d,
		 const PointCloudHelper& pch);

  void publish(const ros::Time& timestamp, 
	       const cv::Mat& colorflags,
	       const PointCloudHelper& pch=PointCloudHelper());
  
};

//////////////////////////////////////////////////////////////////////

ColorHandler::ColorHandler(BlobFinder* b, size_t c):
  parent(b),
  cidx(c),
  active(false),
  num_blobs_subscribers(0),
  num_image_subscribers(0)
  
{
  
  if (cidx >= ColorLUT::numcolors) { return; }
  
  const std::string& name = parent->lut.colornames[cidx];
  
  if (name.empty()) { return; }

  ros::NodeHandle& n = parent->nh;

  blobs_pub = n.advertise<blobfinder::MultiBlobInfo>
    ("/blobfinder/" + name + "/blobs", 100,
     boost::bind(&ColorHandler::subs_inc, 
		 boost::ref(*this), 
		 boost::ref(num_blobs_subscribers), _1),
     boost::bind(&ColorHandler::subs_dec, 
		 boost::ref(*this), 
		 boost::ref(num_blobs_subscribers), _1));

  image_pub = n.advertise<sensor_msgs::Image>
    ("/blobfinder/" + name + "/image", 100,
     boost::bind(&ColorHandler::subs_inc, 
		 boost::ref(*this), 
		 boost::ref(num_image_subscribers), _1),
     boost::bind(&ColorHandler::subs_dec, 
		 boost::ref(*this), 
		 boost::ref(num_image_subscribers), _1));


  blobs3d_pub = n.advertise<blobfinder::MultiBlobInfo3D>
    ("/blobfinder/" + name + "/blobs3d", 100,
     boost::bind(&ColorHandler::subs_inc, 
		 boost::ref(*this), 
		 boost::ref(num_blobs3d_subscribers), _1),
     boost::bind(&ColorHandler::subs_dec, 
		 boost::ref(*this), 
		 boost::ref(num_blobs3d_subscribers), _1));
  
  active = true;
  
}

void ColorHandler::subs_inc(size_t& count,
			    const ros::SingleSubscriberPublisher& ssp) {
  ++count;
}
  
void ColorHandler::subs_dec(size_t& count, 
			    const ros::SingleSubscriberPublisher& ssp) {
  if (count) { --count; }
}

void ColorHandler::get_pos3d(blobfinder::BlobInfo3D& b3d,
			     const PointCloudHelper& pch) {

  assert(pch.ok);

  b3d.have_pos = false;

  int best_dist = 0;
  const int rad = parent->point_search_radius;
  const cv::Rect& rect = parent->roi;
  
  for (int dy=-rad; dy<=rad; ++dy) {
    for (int dx=-rad; dx<=rad; ++dx) {
      
      int dist = dx*dx + dy*dy;

      if (b3d.have_pos && dist >= best_dist) {
	continue;
      }

      int x = b3d.blob.cx + dx + rect.x;
      int y = b3d.blob.cy + dy + rect.y;
      
      if (x < 0 || x >= (int)pch.msg->width ||
	  y < 0 || y >= (int)pch.msg->height) {
	
	continue;
	
      }
	
      const float* xyz = pch.xyz(x,y);
	
      if (isnan(xyz[2])) { 
	continue;
      }
      
      b3d.position.x = xyz[0];
      b3d.position.y = xyz[1];
      b3d.position.z = xyz[2];
      b3d.have_pos = true;
      best_dist = dist;
      
    }
  }



}


void ColorHandler::publish(const ros::Time& timestamp, 
			   const cv::Mat& colorflags,
			   const PointCloudHelper& pch) {
  
  if (!active || 

      !(num_blobs_subscribers ||
	num_image_subscribers ||
	num_blobs3d_subscribers) ) { 

    return; 

  }
  
  const ColorLUT& lut = parent->lut;

  cv::Mat mask;
  lut.colorFlagsToMask(colorflags, cidx, mask);
  
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, 
		   parent->strel, cv::Point(-1,-1), 
		   1, cv::BORDER_REPLICATE);
  
  if (num_image_subscribers) {
    cv_bridge::CvImage cv_img;
    cv_img.image = mask;
    cv_img.encoding = MONO8;
    sensor_msgs::ImagePtr msg = cv_img.toImageMsg();
    msg->header.stamp = timestamp;
    image_pub.publish(msg);
  }
  
  if (num_blobs_subscribers || num_blobs3d_subscribers) {
    
    blobfinder::MultiBlobInfo blobs_msg;
    blobs_msg.header.stamp = timestamp;

    blobfinder::MultiBlobInfo3D blobs3d_msg;
    blobs3d_msg.header.stamp = timestamp;
    
    ColorLUT::RegionInfoVec regions;
    lut.getRegionInfo(mask, regions);
    
    for (size_t i=0; i<regions.size(); ++i) {

      if ( parent->min_blob_area > 0 &&
	   regions[i].area < parent->min_blob_area ) {
	break;
      }

      if ( parent->max_blob_count > 0 &&
	   (int)i >= parent->max_blob_count ) {
	break;
      }
      
      const ColorLUT::RegionInfo& rinfo = regions[i];
      
      blobfinder::BlobInfo binfo;
      
      binfo.area = rinfo.area;
      
      binfo.cx = rinfo.mean.x;
      binfo.cy = rinfo.mean.y;
      
      binfo.ux = rinfo.b1.x;
      binfo.uy = rinfo.b1.y;
      
      binfo.vy = rinfo.b2.y;
      binfo.vx = rinfo.b2.x;
      
      if (num_blobs_subscribers) {
	blobs_msg.blobs.push_back(binfo);
      }

      if (num_blobs3d_subscribers && pch.ok) {
	
	blobfinder::BlobInfo3D binfo3d;
	
	binfo3d.blob = binfo;
	get_pos3d(binfo3d, pch);

	blobs3d_msg.blobs.push_back(binfo3d);

      }
      
    }
    
    if (num_blobs_subscribers) {
      blobs_pub.publish(blobs_msg);
    }

    if (num_blobs3d_subscribers) {
      blobs3d_pub.publish(blobs3d_msg);
    }
    
  }

}

//////////////////////////////////////////////////////////////////////

BlobFinder::BlobFinder() {
  

  int roi_x = 0;
  int roi_y = 0;
  int roi_w = 0;
  int roi_h = 0;
  int open_size = 2;
  bool use_points = false;
  std::string datafile;
  max_blob_count = 0;
  min_blob_area = 0;
  point_search_radius = 3;

  nh.param("/blobfinder/roi_x", roi_x, roi_x);
  nh.param("/blobfinder/roi_y", roi_y, roi_y);
  nh.param("/blobfinder/roi_w", roi_w, roi_w);
  nh.param("/blobfinder/roi_h", roi_h, roi_h);
  nh.param("/blobfinder/open_size", open_size, open_size);
  nh.param("/blobfinder/datafile", datafile, datafile);
  nh.param("/blobfinder/max_blob_count", max_blob_count, max_blob_count);
  nh.param("/blobfinder/min_blob_area", min_blob_area, min_blob_area);
  nh.param("/blobfinder/use_points", use_points, use_points);
  nh.param("/blobfinder/point_search_radius", point_search_radius, point_search_radius);

  int sz = 2*open_size+1;
  
  strel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(sz,sz));

  roi = cv::Rect(roi_x, roi_y, roi_w, roi_h);

  first_image = true;

  lut.load(datafile);

  for (size_t i=0; i<ColorLUT::numcolors; ++i) {
    if (!(lut.colornames[i].empty())) {
      handlers.push_back(new ColorHandler(this, i));
    }
  }

  if (use_points) {
    points_sub = nh.subscribe("points", 4, 
			      &BlobFinder::points_callback, this);
  } else {
    image_sub = nh.subscribe("image", 4, 
			     &BlobFinder::image_callback, this);
  }
  

}


void BlobFinder::setROI(int w, int h) {

  if (roi.x < 0) { roi.x = w + roi.x; }
  if (roi.y < 0) { roi.y = h + roi.y; }
  
  roi.x = std::max(0, std::min(roi.x, w-1));
  roi.y = std::max(0, std::min(roi.y, h-1));
  
  int wr = (w - roi.x);
  int hr = (h - roi.y);
  
  if (roi.width  <= 0) { roi.width  = wr + roi.width;  }
  if (roi.height <= 0) { roi.height = hr + roi.height; }
  
  roi.width =  std::max(1, std::min(roi.width,  wr));
  roi.height = std::max(1, std::min(roi.height, hr));
  
  fprintf(stderr, "ROI is %d, %d, %d, %d\n", 
	  roi.x, roi.y, roi.width, roi.height);
  
  first_image = false;
  
}

  

void BlobFinder::process_image(const cv::Mat& image_rgb, 
			       const PointCloudHelper& pch) {

  if (image_rgb.empty()) { return; }

  if (first_image) {
    setROI(image_rgb.cols, image_rgb.rows);
  }

  cv::Mat image_yuv, colorflags;

  cv::cvtColor(image_rgb, image_yuv, CV_RGB2YCrCb);

  cv::Mat subimage(image_yuv, roi);

  lut.getImageColors(subimage, colorflags);

  ros::Time timestamp = ros::Time::now();

  for (size_t i=0; i<handlers.size(); ++i) {
    handlers[i]->publish(timestamp, colorflags, pch);
  }

}

void BlobFinder::image_callback(const Image::ConstPtr& msg) {

  cv::Mat image_rgb = cv_bridge::toCvCopy(msg, RGB8)->image;

  process_image(image_rgb, 0);

}

void BlobFinder::points_callback(const PointCloud2::ConstPtr& msg) {

  PointCloudHelper pch(msg.get());

  if (!pch.ok) { return; }

  cv::Mat_<cv::Vec3b> image_rgb(msg->height, msg->width);

  for (size_t y=0; y<msg->height; ++y) {
    for (size_t x=0; x<msg->width; ++x) {

      cv::Vec3b dst_rgb;

      const unsigned char* src_rgb = pch.rgb(x,y);

      dst_rgb[2] = src_rgb[0];
      dst_rgb[1] = src_rgb[1];
      dst_rgb[0] = src_rgb[2];

      image_rgb(y,x) = dst_rgb;
      
    }
  }

  process_image(image_rgb, pch);


}

//////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {

  ros::init(argc, argv, "blobfinder");
  
  BlobFinder bf;

  ros::spin();

  return 0;

}
