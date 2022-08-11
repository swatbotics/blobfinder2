#ifndef __COLORLUT_H__
#define __COLORLUT_H__

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/* Color lookup table (LUT) class for detecting blobs of constant
 * color using OpenCV.
 *
 * The BGR colorspace is divided into "bins", each of which contains
 * multiple colors. There are 2^16 = 65536 total bins. 
 *
 * This class a fixed number (see the numcolors enumerant) of named
 * colors.  Each color is represented by the set of bins it covers.
 * To add a color to a bin, you can call the addToColor method.
 *
 * Once colors have been added, you can call the getImageColors method
 * to convert a BGR image into a single channel image, where each
 * pixel is a set of bitflags indicating which colors were matched at
 * each pixel. 
 *
 * There are also convenience methods for turning such a single
 * channel "colorflags" image into per-channel masks, and for
 * extracting statistics (centroid, area, principal components) of
 * connected regions in such masks.
 */
class ColorLUT {
public:

  /* Some helpful enumerants: */
  enum {

    /* Number of bits used to represent the blue, green, and red 
     * channels. Note that if rbits = 4, then there are 16 levels of
     * red.
     */
    bbits = 5,
    gbits = 6,
    rbits = 5,

    /* The number of colors that can be tracked by the lookup table.
     * Note that 8 colors lets us represent all possible color
     * memberships in a single byte. (16 colors would be 2 bytes, 32
     * would be 4 bytes).
     */
    numcolors = 8,

    /* This is an enumerant indicating an invalid color index. 
     * It can be returned by lookupColor or addToColor.
     */
    npos = size_t(-1)
  };

  //////////////////////////////////////////////////////////////////////
  
  /* The colorflags type should be big enough to have at least
   * numcolors bits. The LSB of a colorflags variable corresponds to
   * color 0.
   */
  typedef unsigned char colorflags;

  /* A pixel in a BGR image is a vector of 3 unsigned characters. */
  typedef cv::Vec3b pixel;

  /* Structure to track statistics of a connected component in a mask
   * image. The principal components of a region form the
   * perpendicular major and minor axes of an ellipse that
   * approximates the shape of the region.
   */
  struct RegionInfo {

    /* Area in pixels. */
    float area;

    /* Centroid of the region. */
    cv::Point2f mean;

    /* First (larger) principal component vector. */
    cv::Point2f b1;

    /* Second (smaller) principal component vector. */
    cv::Point2f b2;
  };
                                                        
  /* Typedef for a vector of RegionInfo. */
  typedef std::vector<RegionInfo> RegionInfoVec;

  /* Typedef for vector of OpenCV points. */
  typedef std::vector<cv::Point>  PointVec;

  /* Typedef for vector of vector of OpenCV points. */
  typedef std::vector<PointVec>   PointVecs;
  
  //////////////////////////////////////////////////////////////////////

  /* This vector holds, for each bin, the colors to which the bin
   * belongs. Rather than accessing it directly, you should use
   * the addToColor and removeFromColor methods below.
   */
  std::vector<colorflags> lutdata;

  /* Array of strings indicating the name for each color.  Empty
   * strings are assumed to be unused spots.
   */
  std::string colornames[numcolors];

  //////////////////////////////////////////////////////////////////////

  /* Constructor. */
  ColorLUT();

  /* Add the bin contianing the given pixel color to be a member of the
   * color whose index is provided.  You can also optionally provide
   * ranges to add in surrounding bins (for instance, yRange=1 would
   * mean to also add the two adjacent bins above and below the
   * current bin in luminance).
   */
  void addToColor(const pixel& bgr, size_t cidx);
  
  /* Remove the bin containing the given pixel color from membership
   * in the color whose index is provided.  You can also optionally
   * provide ranges to remove surrounding bins (for instance, yRange=1
   * would mean to also remove the two adjacent bins above and below
   * the current bin in luminance).
   */
  void removeFromColor(const pixel& bgr, size_t cidx);

  /* Get the colors for which the bin containing the given pixel
   * color is a member. 
   */
  colorflags getColors(const pixel& bgr) const;

  /* Remove all bins from membership in the color whose index is
   * provided. This is called automatically by removeFromColor.
   */
  void clearColor(size_t cidx);

  /* Add a new color with the given name to the next unused spot.
   * If there are no unused spots, this returns npos. Adding colors
   * will not alter the indices of other existing colors.
   */
  size_t addColor(const std::string& name);

  /* Remove the color with the given index and set that color's spot
   * to be unused.  Also calls clearColor on the color to remove
   * all bins from membership in that color.  Removing colors 
   * will not alter the indices of other existing colors.
   */
  void removeColor(size_t cidx);

  /* Return the index of the color whose name matches the one
   * provided, or npos if none was found.
   */
  size_t lookupColor(const std::string& name) const;

  /* Save this LUT to a file with the name provided. */
  void save(const std::string& filename) const;

  /* Load an LUT from a file with the name provided. */
  void load(const std::string& filename);

  /* This takes a BGR image as input and outputs a single-channel
   * colorflags image as output, where each pixel in the output has
   * bits corresponding to the color membership of the corresponding
   * input pixel.
   */
  void getImageColors(const cv::Mat& image, 
		      cv::Mat& colorflags) const;

  
  /* This convenience function takes as input a single-channel
   * colorflags image and a color index, and sets each pixel of the
   * 8-bit single channel output mask to a non-zero value if the
   * corresponding input pixel was a member of the color provided.
   */
  void colorFlagsToMask(const cv::Mat& colorflags, 
			size_t cidx,
			cv::Mat& mask) const;

  /* This takes a BGR image and a color index as input and outputs a
   * single-channel mask image as output, where each pixel in the
   * output is non-zero if the corresponding input pixel was a member
   * of the color provided.  Note that since this internally calls
   * getImageColors(), it is not particularly efficient to call this
   * method multiple times in a row.  Instead, make a single call to
   * getImageColors() and multiple calls to colorFlagsToMask().
   */
  void getImageColor(const cv::Mat& image, 
		     size_t cidx, 
		     cv::Mat& mask) const;


  /* This function takes a single-channel mask image as input, and
   * outputs a vector of vectors of points.  Each inner vector of points
   * is a contour or outline of a connected region in the mask image.
   */
  void getContours(const cv::Mat& mask,
                   PointVecs& regions) const;

  /* Computes the region info (centroid, area, etc.) of the contour
   * given as input.
   */
  void contourToRegionInfo(const PointVec& region,
                           RegionInfo& info) const;

  /* Returns a RegionInfo struct for each connected component in
   * the input mask image.
   */
  void getRegionInfo(const cv::Mat& mask, 
		     RegionInfoVec& info) const;

  /* Gets the mean of every color or (127, 127, 127) if empty. */
  void getMeanColors(pixel mean_colors[numcolors]) const;

};

#endif

