// A node for 
//
// config/arguments:
//   - input image (eg /camera/bgr/image_raw)
//   - ROI (default whole window, otherwise x,y,w,h)
//   - minimum blob size (default 1)
//   - # blobs
//
// outputs
//   - /blobfinder2/blobs
//     - centroid, area, and principal component of each detected blob for color_foo
//   - todo doc more


#include "ros/ros.h"
#include <blobfinder2/MultiBlobInfo.h>
#include <blobfinder2/MultiBlobInfo3D.h>
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
using sensor_msgs::image_encodings::BGR8;
using sensor_msgs::image_encodings::MONO8;

using sensor_msgs::PointCloud2;
using sensor_msgs::PointField;

//////////////////////////////////////////////////////////////////////
// We will define a class later to handle each color separately.

//class ColorHandler;

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

class BlobFinder2 {
public:

  // node handle for this node
  ros::NodeHandle nh;

  // lookup table being used
  ColorLUT lut;

  // mean colors per channel
  cv::Vec3b mean_colors[ColorLUT::numcolors];

  // how many named colors are there in the LUT?
  size_t num_named_colors;

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

  // counters for various subscribers
  size_t num_blobs_subscribers;
  size_t num_blobs3d_subscribers;
  size_t num_colorflags_subscribers;
  size_t num_debug_image_subscribers;

  // publishers for outputs
  ros::Publisher blobs_pub;
  ros::Publisher blobs3d_pub;
  ros::Publisher colorflags_pub;
  ros::Publisher debug_image_pub;


  // constructor
  BlobFinder2();

  // image callback
  void image_callback(const Image::ConstPtr& msg);

  // point cloud callback
  void points_callback(const PointCloud2::ConstPtr& msg);

  void setROI(int w, int h);
  void process_image(const cv::Mat& image_bgr, 
		     const PointCloudHelper& pch=PointCloudHelper());

  void subs_inc(size_t&, const ros::SingleSubscriberPublisher&);
  void subs_dec(size_t&, const ros::SingleSubscriberPublisher&);

  void get_pos3d(blobfinder2::BlobInfo3D& b3d,
		 const PointCloudHelper& pch);
  
};

/*
//////////////////////////////////////////////////////////////////////
// Class for handling one color of a BlobFinder2

class ColorHandler {
public:
  
  BlobFinder2* parent;

  size_t cidx;
  
  bool active;

  size_t num_blobs_subscribers;
  size_t num_image_subscribers;
  size_t num_blobs3d_subscribers;
  
  ros::Publisher blobs_pub;
  ros::Publisher image_pub;
  ros::Publisher blobs3d_pub;

  ColorHandler(BlobFinder2* b, size_t cidx);

  void subs_inc(size_t&, const ros::SingleSubscriberPublisher& ssp);
  void subs_dec(size_t&, const ros::SingleSubscriberPublisher& ssp);

  void get_pos3d(blobfinder2::BlobInfo3D& b3d,
		 const PointCloudHelper& pch);

  void publish(const ros::Time& timestamp, 
	       const cv::Mat& colorflags,
	       const PointCloudHelper& pch=PointCloudHelper());
  
};

//////////////////////////////////////////////////////////////////////

ColorHandler::ColorHandler(BlobFinder2* b, size_t c):
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

  blobs_pub = n.advertise<blobfinder2::MultiBlobInfo>
    ("/blobfinder2/" + name + "/blobs", 100,
     boost::bind(&ColorHandler::subs_inc, 
		 boost::ref(*this), 
		 boost::ref(num_blobs_subscribers), _1),
     boost::bind(&ColorHandler::subs_dec, 
		 boost::ref(*this), 
		 boost::ref(num_blobs_subscribers), _1));

  image_pub = n.advertise<sensor_msgs::Image>
    ("/blobfinder2/" + name + "/image", 100,
     boost::bind(&ColorHandler::subs_inc, 
		 boost::ref(*this), 
		 boost::ref(num_image_subscribers), _1),
     boost::bind(&ColorHandler::subs_dec, 
		 boost::ref(*this), 
		 boost::ref(num_image_subscribers), _1));


  blobs3d_pub = n.advertise<blobfinder2::MultiBlobInfo3D>
    ("/blobfinder2/" + name + "/blobs3d", 100,
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

void ColorHandler::get_pos3d(blobfinder2::BlobInfo3D& b3d,
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
    
    blobfinder2::MultiBlobInfo blobs_msg;
    blobs_msg.header.stamp = timestamp;

    blobfinder2::MultiBlobInfo3D blobs3d_msg;
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
      
      blobfinder2::BlobInfo binfo;
      
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
	
	blobfinder2::BlobInfo3D binfo3d;
	
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
*/

//////////////////////////////////////////////////////////////////////

BlobFinder2::BlobFinder2() {
  

  int roi_x = 0;
  int roi_y = 0;
  int roi_w = 0;
  int roi_h = 0;

  bool use_points = false;
  std::string datafile;
  max_blob_count = 0;
  min_blob_area = 0;
  point_search_radius = 3;

  nh.param("/blobfinder2/roi_x", roi_x, roi_x);
  nh.param("/blobfinder2/roi_y", roi_y, roi_y);
  nh.param("/blobfinder2/roi_w", roi_w, roi_w);
  nh.param("/blobfinder2/roi_h", roi_h, roi_h);
  nh.param("/blobfinder2/datafile", datafile, datafile);
  nh.param("/blobfinder2/max_blob_count", max_blob_count, max_blob_count);
  nh.param("/blobfinder2/min_blob_area", min_blob_area, min_blob_area);
  nh.param("/blobfinder2/use_points", use_points, use_points);
  nh.param("/blobfinder2/point_search_radius", point_search_radius, point_search_radius);


  roi = cv::Rect(roi_x, roi_y, roi_w, roi_h);

  first_image = true;

  fprintf(stderr, "loading datafile %s...\n", datafile.c_str());

  lut.load(datafile);

  lut.getMeanColors(mean_colors);

  num_named_colors = 0;

  for (size_t cidx=0; cidx<ColorLUT::numcolors; ++cidx) {
    
    if (lut.colornames[cidx] != "") {

      num_named_colors = cidx + 1;
      
      const cv::Vec3b& mc = mean_colors[cidx];
      
      fprintf(stderr, "mean color for %s is (%d, %d, %d)\n",
	      lut.colornames[cidx].c_str(),
	      int(mc[0]), int(mc[1]), int(mc[2]));
      
    }
    
  }

  /*
  for (size_t i=0; i<ColorLUT::numcolors; ++i) {
    if (!(lut.colornames[i].empty())) {
      handlers.push_back(new ColorHandler(this, i));
    }
  }
  */

  num_blobs_subscribers = 0;
  num_blobs3d_subscribers = 0;
  num_colorflags_subscribers = 0;
  num_debug_image_subscribers = 0;

  int qsz = 4;

  blobs_pub = nh.advertise<blobfinder2::MultiBlobInfo>
    ("/blobfinder2/blobs", qsz,
     boost::bind(&BlobFinder2::subs_inc,
		 boost::ref(*this),
		 boost::ref(num_blobs_subscribers), _1),
     boost::bind(&BlobFinder2::subs_dec,
		 boost::ref(*this),
		 boost::ref(num_blobs_subscribers), _1));

    blobs3d_pub = nh.advertise<blobfinder2::MultiBlobInfo3D>
    ("/blobfinder2/blobs3d", qsz,
     boost::bind(&BlobFinder2::subs_inc,
		 boost::ref(*this),
		 boost::ref(num_blobs3d_subscribers), _1),
     boost::bind(&BlobFinder2::subs_dec,
		 boost::ref(*this),
		 boost::ref(num_blobs3d_subscribers), _1));

  colorflags_pub = nh.advertise<sensor_msgs::Image>
    ("/blobfinder2/colorflags", qsz,
     boost::bind(&BlobFinder2::subs_inc,
		 boost::ref(*this),
		 boost::ref(num_colorflags_subscribers), _1),
     boost::bind(&BlobFinder2::subs_dec,
		 boost::ref(*this),
		 boost::ref(num_colorflags_subscribers), _1));

  debug_image_pub = nh.advertise<sensor_msgs::Image>
    ("/blobfinder2/debug_image", qsz,
     boost::bind(&BlobFinder2::subs_inc,
		 boost::ref(*this),
		 boost::ref(num_debug_image_subscribers), _1),
     boost::bind(&BlobFinder2::subs_dec,
		 boost::ref(*this),
		 boost::ref(num_debug_image_subscribers), _1));

  if (use_points) {
    points_sub = nh.subscribe("points", 4, 
			      &BlobFinder2::points_callback, this);
  } else {
    image_sub = nh.subscribe("image", 4, 
			     &BlobFinder2::image_callback, this);
  }

  
  

}


void BlobFinder2::setROI(int w, int h) {

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

void BlobFinder2::subs_inc(size_t& count,
			   const ros::SingleSubscriberPublisher& _) {
  ++count;
  fprintf(stderr, "after subs_inc, count is %d\n", (int)count);
}

void BlobFinder2::subs_dec(size_t& count,
			   const ros::SingleSubscriberPublisher& _) {
  if (count) { --count; }
  fprintf(stderr, "after subs_dec, count is %d\n", (int)count);
}


void BlobFinder2::get_pos3d(blobfinder2::BlobInfo3D& b3d,
			    const PointCloudHelper& pch) {

  assert(pch.ok);

  b3d.have_pos = false;

  int best_dist = 0;
  const int rad = point_search_radius;
  const cv::Rect& rect = roi;
  
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

void BlobFinder2::process_image(const cv::Mat& image_bgr, 
			       const PointCloudHelper& pch) {

  if (image_bgr.empty()) { return; }

  if (first_image) {
    setROI(image_bgr.cols, image_bgr.rows);
  }

  if (num_blobs_subscribers == 0 &&
      num_blobs3d_subscribers == 0 &&
      num_debug_image_subscribers == 0 &&
      num_colorflags_subscribers == 0) {
    // TODO: pause image/points sub?
    return;
  }

  ros::Time timestamp = ros::Time::now();

  cv::Mat colorflags;

  cv::Mat subimage(image_bgr, roi);

  lut.getImageColors(subimage, colorflags);

  ros::Time pre_colorflags_pub = ros::Time::now();


  if (num_colorflags_subscribers > 0) {

    cv_bridge::CvImage cv_img;
    cv_img.image = colorflags;
    cv_img.encoding = MONO8;
    sensor_msgs::ImagePtr msg = cv_img.toImageMsg();
    msg->header.stamp = timestamp;

    colorflags_pub.publish(msg);

  }

  if (num_blobs_subscribers == 0 &&
      num_blobs3d_subscribers == 0 &&
      num_debug_image_subscribers == 0) {
    // TODO: pause image/points sub?
    return;
  }

  cv::Mat cidx_masks_3d, debug_mask, debug_image_bgr;
  
  int rows_cols_channels[3] = {
    subimage.rows,
    subimage.cols,
    num_named_colors
  };

  ros::Time pre_mask_split = ros::Time::now();


  cidx_masks_3d = cv::Mat(3, rows_cols_channels, CV_8U);

  // split masks
  {

    cv::Mat cidx_masks_2d = cidx_masks_3d.reshape(num_named_colors,
						  2, rows_cols_channels);

    cv::Mat cidx_mask;
    cv::Mat one = cv::Mat::ones(1, 1, CV_8U);

    for (size_t cidx=0; cidx<num_named_colors; ++cidx) {

      // and colorflags with 1 to get channel
      cv::bitwise_and(colorflags, one, cidx_mask);

      // change 0/1 to 0/255
      cidx_mask *= 255;

      // divide colorflags by 2 to shift to the right
      colorflags /= 2;

      // insert into 2d
      cv::insertChannel(cidx_mask, cidx_masks_2d, cidx);
      
      
    }

  }

  ros::Time pre_cleanup = ros::Time::now();

  if (lut.mask_blur_sigma > 0.0) {
 
    cv::Mat cidx_masks_2d = cidx_masks_3d.reshape(num_named_colors,
						  2, rows_cols_channels);

    // clean it up
    cv::GaussianBlur(cidx_masks_2d, cidx_masks_2d,
		     cv::Size(0, 0), lut.mask_blur_sigma);
    
    // re-binarize after blur
    cv::threshold(cidx_masks_2d, cidx_masks_2d,
		  127, 255, cv::THRESH_BINARY);
   

  }

  ros::Time pre_debug_image = ros::Time::now();

  if (num_debug_image_subscribers > 0) {

    cv::Mat debug_image_bgr = cv::Mat::zeros(subimage.rows, subimage.cols, CV_8UC3);

    cv::Mat cidx_masks_2d = cidx_masks_3d.reshape(num_named_colors,
						  2, rows_cols_channels);

    cv::Mat cidx_mask;

    for (size_t i=0; i<num_named_colors; ++i) {

      size_t cidx = num_named_colors - i - 1;

      cv::extractChannel(cidx_masks_2d, cidx_mask, cidx);

      cv::Mat color = cv::Mat(3, 1, CV_8U, mean_colors + cidx);

      debug_image_bgr.setTo(color, cidx_mask);
      
    }

    cv_bridge::CvImage cv_img;
    cv_img.image = debug_image_bgr;
    cv_img.encoding = BGR8;
    sensor_msgs::ImagePtr msg = cv_img.toImageMsg();
    msg->header.stamp = timestamp;

    debug_image_pub.publish(msg);

  }

  ros::Time pre_blob = ros::Time::now();

  if (num_blobs_subscribers || num_blobs3d_subscribers) {

    int rows_cols[2] = { subimage.rows, subimage.cols };

    cv::Mat cidx_masks_2d = cidx_masks_3d.reshape(num_named_colors,
						  2, rows_cols);

    cv::Mat cidx_mask;

    blobfinder2::MultiBlobInfo blobs_msg;
    blobs_msg.header.stamp = timestamp;

    blobfinder2::MultiBlobInfo3D blobs3d_msg;
    blobs3d_msg.header.stamp = timestamp;

    for (size_t cidx=0; cidx<num_named_colors; ++cidx) {

      if (lut.colornames[cidx] == "") { continue; }

      cv::extractChannel(cidx_masks_2d, cidx_mask, cidx);

      ColorLUT::RegionInfoVec regions;
      lut.getRegionInfo(cidx_mask, regions);

      blobfinder2::ColorBlobInfo color_blob;

      blobfinder2::ColorBlobInfo3D color_blob3d;

      color_blob.color.data = lut.colornames[cidx];

      color_blob3d.color.data = lut.colornames[cidx];
    
      for (size_t i=0; i<regions.size(); ++i) {

	if (regions[i].area == 0.0) {
	  continue;
	}

	if ( min_blob_area > 0 && regions[i].area < min_blob_area ) {
	  break;
	}

	if ( max_blob_count > 0 && (int)i >= max_blob_count ) {
	  break;
	}
      
	const ColorLUT::RegionInfo& rinfo = regions[i];
      
	blobfinder2::BlobInfo binfo;
      
	binfo.area = rinfo.area;
      
	binfo.cx = rinfo.mean.x;
	binfo.cy = rinfo.mean.y;
      
	binfo.ux = rinfo.b1.x;
	binfo.uy = rinfo.b1.y;
      
	binfo.vy = rinfo.b2.y;
	binfo.vx = rinfo.b2.x;
      
	if (num_blobs_subscribers) {
	  
	  color_blob.blobs.push_back(binfo);
	  
	}

	if (num_blobs3d_subscribers) {

	  blobfinder2::BlobInfo3D binfo3d;
	  
	  binfo3d.blob = binfo;
	  
	  if (pch.ok) {

	    get_pos3d(binfo3d, pch);

	  } else {

	    binfo3d.position.x = 0.0;
	    binfo3d.position.y = 0.0;
	    binfo3d.position.z = 0.0;
	  
	    binfo3d.have_pos = false;

	  }

	  color_blob3d.blobs.push_back(binfo3d);
	  
	}
      
      }

      if (num_blobs_subscribers) {
	blobs_msg.color_blobs.push_back(color_blob);
      }

      if (num_blobs3d_subscribers) {
	blobs3d_msg.color_blobs.push_back(color_blob3d);
      }
      
    }
    
    if (num_blobs_subscribers) {
      blobs_pub.publish(blobs_msg);
    }

    if (num_blobs3d_subscribers) {
      blobs3d_pub.publish(blobs3d_msg);
    }
    
  }

  ros::Time finish = ros::Time::now();

  fprintf(stderr, "  get flags:      %.4fs\n", (pre_colorflags_pub - timestamp).toSec());
  fprintf(stderr, "  mask split:     %.4fs\n", (pre_cleanup-pre_mask_split).toSec());
  fprintf(stderr, "  cleanup:        %.4fs\n", (pre_debug_image-pre_cleanup).toSec());
  fprintf(stderr, "  debug image:    %.4fs\n", (pre_blob-pre_debug_image).toSec());
  fprintf(stderr, "  blob message:   %.4fs\n", (finish-pre_blob).toSec());
  fprintf(stderr, "total processing: %.4fs\n\n", (finish-timestamp).toSec());
  

  /*
  for (size_t i=0; i<handlers.size(); ++i) {
    handlers[i]->publish(timestamp, colorflags, pch);
  }
  */

}

void BlobFinder2::image_callback(const Image::ConstPtr& msg) {

  cv::Mat image_bgr = cv_bridge::toCvCopy(msg, BGR8)->image;

  process_image(image_bgr, 0);

}

void BlobFinder2::points_callback(const PointCloud2::ConstPtr& msg) {

  PointCloudHelper pch(msg.get());

  if (!pch.ok) { return; }

  cv::Mat_<cv::Vec3b> image_bgr(msg->height, msg->width);

  for (size_t y=0; y<msg->height; ++y) {
    for (size_t x=0; x<msg->width; ++x) {

      cv::Vec3b dst_bgr;

      const unsigned char* src_bgr = pch.rgb(x,y);

      dst_bgr[0] = src_bgr[0];
      dst_bgr[1] = src_bgr[1];
      dst_bgr[2] = src_bgr[2];

      image_bgr(y,x) = dst_bgr;
      
    }
  }

  process_image(image_bgr, pch);


}

//////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {

  ros::init(argc, argv, "blobfinder2");
  
  BlobFinder2 bf;

  ros::spin();

  return 0;

}
